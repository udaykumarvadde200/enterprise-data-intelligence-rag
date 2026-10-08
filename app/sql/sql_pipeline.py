from app.db.connection import DatabaseConnection
from app.db.schema import DatabaseSchemaInspector
from app.llm.ollama_client import OllamaClient
from app.sql.ambiguity_checker import AmbiguityChecker
from app.sql.customer_reference_extractor import CustomerReferenceExtractor
from app.sql.schema_formatter import SchemaFormatter
from app.sql.sql_executor import SQLExecutor
from app.sql.sql_generator import SQLGenerator
from app.sql.sql_validator import SQLValidator
from app.conversation.state import ConversationState
from app.sql.clarification_resolver import ClarificationResolver


class SQLPipeline:
    def __init__(self):
        self.db = DatabaseConnection()

        self.schema_inspector = DatabaseSchemaInspector(self.db)
        self.schema_formatter = SchemaFormatter()

        self.llm = OllamaClient()

        self.sql_generator = SQLGenerator(self.llm)
        self.sql_validator = SQLValidator()
        self.sql_executor = SQLExecutor(self.db)

        self.customer_reference_extractor = (
            CustomerReferenceExtractor(self.llm)
        )

        self.ambiguity_checker = AmbiguityChecker(self.db)

        self.conversation_state = ConversationState()
        self.clarification_resolver = ClarificationResolver()

    def run(self, question: str):

        # Stores the exact customer ID when a clarification
        # resolves an ambiguous customer reference.
        resolved_customer_id = None

        # ---------------------------------------------------------
        # 1. Handle response to a previous clarification
        # ---------------------------------------------------------
        if self.conversation_state.status == "awaiting_clarification":

            resolution = self.clarification_resolver.resolve_customer(
                question,
                self.conversation_state.pending_customers,
            )

            if resolution["status"] == "not_found":
                return {
                    "status": "clarification_required",
                    "message": (
                        "I could not determine which customer you mean. "
                        "Please specify the city or full customer name."
                    ),
                }

            if resolution["status"] == "ambiguous":

                options = "\n".join(
                    f"- {name} — {city}"
                    for _customer_id, name, city
                    in resolution["customers"]
                )

                return {
                    "status": "clarification_required",
                    "message": (
                        "Your clarification still matches "
                        "multiple customers:\n"
                        f"{options}"
                    ),
                }

            # Customer successfully resolved
            customer_id, customer_name, customer_city = (
                resolution["customer"]
            )

            resolved_customer_id = customer_id

            # Recover original question
            original_question = (
                self.conversation_state.pending_question
            )

            pending_reference = (
                self.conversation_state.pending_customer_reference
            )

            # Clear clarification state
            self.conversation_state.status = None
            self.conversation_state.pending_question = None
            self.conversation_state.pending_customer_reference = None
            self.conversation_state.pending_customers = []

            # Replace the ambiguous reference with the resolved
            # customer's name for natural-language context.
            question = original_question.replace(
                pending_reference,
                customer_name,
            )

        # ---------------------------------------------------------
        # 2. Extract customer reference
        # ---------------------------------------------------------
        customer_reference = (
            self.customer_reference_extractor.extract(question)
        )

        # ---------------------------------------------------------
        # 3. Check customer ambiguity
        #
        # Skip this if the customer was already resolved through
        # the clarification flow.
        # ---------------------------------------------------------
        if (
            customer_reference != "NONE"
            and resolved_customer_id is None
        ):

            ambiguity_result = (
                self.ambiguity_checker.check_customer_name(
                    customer_reference
                )
            )

            if ambiguity_result["status"] == "ambiguous":

                matches = ambiguity_result["matches"]

                self.conversation_state.pending_question = question
                self.conversation_state.pending_customer_reference = (
                    customer_reference
                )
                self.conversation_state.pending_customers = matches
                self.conversation_state.status = (
                    "awaiting_clarification"
                )

                options = "\n".join(
                    f"- {name} — {city}"
                    for _customer_id, name, city in matches
                )

                return {
                    "status": "clarification_required",
                    "message": (
                        f"Multiple customers match "
                        f"'{customer_reference}'.\n"
                        f"Please specify which customer you mean:\n"
                        f"{options}"
                    ),
                }

            if ambiguity_result["status"] == "not_found":

                return {
                    "status": "not_found",
                    "message": (
                        f"No customer was found matching "
                        f"'{customer_reference}'."
                    ),
                }

        # ---------------------------------------------------------
        # 4. Get database schema
        # ---------------------------------------------------------
        schema = self.schema_inspector.get_schema()

        # ---------------------------------------------------------
        # 5. Format schema
        # ---------------------------------------------------------
        formatted_schema = (
            self.schema_formatter.format_schema(schema)
        )

        # ---------------------------------------------------------
        # 6. Generate SQL
        # ---------------------------------------------------------
        sql = self.sql_generator.generate(
            question,
            formatted_schema,
            resolved_customer_id=resolved_customer_id,
        )

        # ---------------------------------------------------------
        # 7. Validate SQL syntax
        # ---------------------------------------------------------
        if not self.sql_validator.validate_syntax(sql):
            raise ValueError(
                "Generated SQL has invalid syntax."
            )

        # ---------------------------------------------------------
        # 8. Validate database schema
        # ---------------------------------------------------------
        if not self.sql_validator.validate_schema(
            sql,
            schema,
        ):
            raise ValueError(
                "Generated SQL references an invalid schema."
            )

        # ---------------------------------------------------------
        # 9. Validate read-only safety
        # ---------------------------------------------------------
        if not self.sql_validator.validate_read_only(sql):
            raise ValueError(
                "Only read-only SQL queries are allowed."
            )

        # ---------------------------------------------------------
        # 10. Execute SQL
        # ---------------------------------------------------------
        results = self.sql_executor.execute(sql)

        return {
            "status": "success",
            "sql": sql,
            "results": results,
        }