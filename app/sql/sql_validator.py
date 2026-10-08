import sqlglot
from sqlglot import exp


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

        query_tables = {}

        for table in parsed_sql.find_all(exp.Table):
            table_name = table.name
            alias = table.alias_or_name

            if table_name not in valid_schema:
                return False

            query_tables[alias] = table_name

        for column in parsed_sql.find_all(exp.Column):
            column_name = column.name
            table_reference = column.table

            if table_reference:
                if table_reference not in query_tables:
                    return False

                real_table = query_tables[table_reference]

                if column_name not in valid_schema[real_table]:
                    return False

            else:
                matching_tables = [
                    table_name
                    for table_name in query_tables.values()
                    if column_name in valid_schema[table_name]
                ]

                if not matching_tables:
                    return False

                if len(set(matching_tables)) > 1:
                    return False

        return True

    def validate_read_only(self, sql: str) -> bool:
        try:
            parsed_sql = sqlglot.parse_one(sql, dialect="postgres")
        except sqlglot.errors.ParseError:
            return False

        return isinstance(parsed_sql, exp.Select)