import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope


class SQLValidator:
    # Statement types that must never appear anywhere in an accepted query,
    # including inside CTEs and subqueries (e.g. WITH d AS (DELETE ...)).
    _WRITE_NODE_NAMES = (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Copy",
        "TruncateTable",
        "Set",
        "Analyze",
    )
    WRITE_NODES = tuple(
        getattr(exp, name)
        for name in _WRITE_NODE_NAMES
        if hasattr(exp, name)
    )

    # sqlglot parses functions it does not model as exp.Anonymous. Those are
    # rejected unless allowlisted. This blocks pg_sleep, pg_read_file,
    # nextval, lo_import, dblink, set_config and similar side-effecting or
    # file/network-reaching functions.
    ALLOWED_ANONYMOUS_FUNCTIONS = frozenset({"age"})

    def validate_syntax(self, sql: str) -> bool:
        try:
            sqlglot.parse_one(sql, dialect="postgres")
            return True
        except SqlglotError:
            # ParseError and TokenError (e.g. prose containing an
            # apostrophe) must both be reported as invalid SQL.
            return False

    def validate_schema(self, sql: str, schema: dict) -> bool:
        try:
            parsed_sql = sqlglot.parse_one(sql, dialect="postgres")
            scopes = traverse_scope(parsed_sql)
        except SqlglotError:
            return False

        valid_schema = {}

        for (
            table_name,
            column_name,
            _data_type,
            _is_nullable,
            _default,
            _position,
        ) in schema["columns"]:
            valid_schema.setdefault(table_name, set()).add(column_name)

        for scope in scopes:
            sources = scope.sources

            # Validate every real table used by this query scope.
            for source in sources.values():
                if isinstance(source, exp.Table):
                    if source.name not in valid_schema:
                        return False

            # Validate only columns belonging to this SELECT,
            # excluding columns inside nested SELECT expressions.
            for column in (
                column
                for column in scope.expression.find_all(exp.Column)
                if column.find_ancestor(exp.Select) is scope.expression
            ):
                if column.name == "*":
                    continue

                column_name = column.name
                table_reference = column.table

                if table_reference:
                    source = sources.get(table_reference)

                    if source is None:
                        return False

                    if isinstance(source, exp.Table):
                        if column_name not in valid_schema.get(
                            source.name, set()
                        ):
                            return False

                    elif isinstance(source, type(scope)):
                        # CTE or derived-table source.
                        output_columns = {
                            item.alias_or_name
                            for item in source.expression.selects
                        }
                        if column_name not in output_columns:
                            return False
                    else:
                        return False

                else:
                    matching_sources = []

                    for source in sources.values():
                        if isinstance(source, exp.Table):
                            if column_name in valid_schema.get(
                                source.name, set()
                            ):
                                matching_sources.append(source.name)

                        elif isinstance(source, type(scope)):
                            output_columns = {
                                item.alias_or_name
                                for item in source.expression.selects
                            }
                            if column_name in output_columns:
                                matching_sources.append(source)

                    # Reject missing or ambiguous unqualified columns.
                    if len(matching_sources) != 1:
                        return False

        return True

    def validate_read_only(self, sql: str) -> bool:
        try:
            statements = sqlglot.parse(sql, dialect="postgres")
        except SqlglotError:
            return False

        # Require exactly one non-empty SQL statement.
        if len(statements) != 1:
            return False

        parsed_sql = statements[0]

        # Only ordinary SELECT queries are allowed.
        if not isinstance(parsed_sql, exp.Select):
            return False

        # SELECT INTO creates a table; reject it.
        if parsed_sql.args.get("into") is not None:
            return False

        # Reject row-locking clauses such as FOR UPDATE.
        if parsed_sql.args.get("locks"):
            return False

        # Reject data-modifying statements nested anywhere in the query.
        if self.WRITE_NODES and parsed_sql.find(*self.WRITE_NODES):
            return False

        # Reject functions sqlglot does not recognize, unless allowlisted.
        for function in parsed_sql.find_all(exp.Anonymous):
            if function.name.lower() not in self.ALLOWED_ANONYMOUS_FUNCTIONS:
                return False

        return True