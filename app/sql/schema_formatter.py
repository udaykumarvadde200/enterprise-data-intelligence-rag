class SchemaFormatter:
    def format_schema(self, schema):
        lines = []

        tables = schema["tables"]
        columns = schema["columns"]
        primary_keys = schema["primary_keys"]
        foreign_keys = schema["foreign_keys"]

        for table_name in tables:
            lines.append(f"TABLE {table_name}")

            # Columns
            lines.append("COLUMNS:")

            table_columns = [
                column
                for column in columns
                if column[0] == table_name
            ]

            for (
                _table_name,
                column_name,
                data_type,
                is_nullable,
                column_default,
                _ordinal_position,
            ) in table_columns:

                nullable = "NULL" if is_nullable == "YES" else "NOT NULL"

                lines.append(
                    f"  - {column_name} "
                    f"({data_type}, {nullable})"
                )

            # Primary keys
            table_primary_keys = [
                column_name
                for table, column_name in primary_keys
                if table == table_name
            ]

            if table_primary_keys:
                lines.append(
                    f"PRIMARY KEY: {', '.join(table_primary_keys)}"
                )

            # Foreign keys
            table_foreign_keys = [
                foreign_key
                for foreign_key in foreign_keys
                if foreign_key[0] == table_name
            ]

            if table_foreign_keys:
                lines.append("FOREIGN KEYS:")

                for (
                    _table_name,
                    column_name,
                    referenced_table,
                    referenced_column,
                ) in table_foreign_keys:

                    lines.append(
                        f"  - {column_name} → "
                        f"{referenced_table}.{referenced_column}"
                    )

            lines.append("")

        return "\n".join(lines)