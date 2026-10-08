class ClarificationResolver:
    def resolve_customer(self, clarification: str, candidates: list):
        clarification = clarification.lower().strip()

        matches = []

        for customer_id, name, city in candidates:
            city_lower = city.lower()
            name_lower = name.lower()

            if (
                city_lower in clarification
                or name_lower in clarification
            ):
                matches.append(
                    (customer_id, name, city)
                )

        if len(matches) == 1:
            return {
                "status": "resolved",
                "customer": matches[0],
            }

        if len(matches) > 1:
            return {
                "status": "ambiguous",
                "customers": matches,
            }

        return {
            "status": "not_found",
            "customers": [],
        }