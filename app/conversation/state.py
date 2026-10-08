class ConversationState:
    def __init__(self):
        self.pending_question = None
        self.pending_customer_reference = None
        self.pending_customers = []
        self.status = None