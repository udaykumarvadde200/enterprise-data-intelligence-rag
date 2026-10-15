import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope


class SQLValidator:
    def validate_syntax(self, sql: str) -> bool:
        try:
            sqlglot.parse_one(sql, dialect="postgres")
            return True
        except sqlglot.errors.ParseError:
            return False

    def validate_schema(self, sql: str, schema: dict) -> bool:
        try:
            parsed_sql = sqlglot.parse_one(sql, dialect="postgres")
        except sqlglot.errors.ParseError:
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

        for scope in traverse_scope(parsed_sql):
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
        except sqlglot.errors.ParseError:
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

        return True