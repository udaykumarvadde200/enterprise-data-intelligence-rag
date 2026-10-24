from unittest.mock import patch

from app.graph.workflow import EnterpriseWorkflow


def test_sql_pipeline_isolated_by_user_and_thread():
    workflow = EnterpriseWorkflow()

    with patch("app.graph.workflow.SQLPipeline") as mock_pipeline:
        # Return a fresh mock instance for every new pipeline.
        mock_pipeline.side_effect = [
            object(),
            object(),
            object(),
        ]

        user_a_thread_1 = workflow._get_sql_pipeline(
            thread_id="thread-1",
            user_id="user-a",
        )
        user_a_thread_1_again = workflow._get_sql_pipeline(
            thread_id="thread-1",
            user_id="user-a",
        )
        user_b_thread_1 = workflow._get_sql_pipeline(
            thread_id="thread-1",
            user_id="user-b",
        )
        user_a_thread_2 = workflow._get_sql_pipeline(
            thread_id="thread-2",
            user_id="user-a",
        )

    assert user_a_thread_1 is user_a_thread_1_again
    assert user_a_thread_1 is not user_b_thread_1
    assert user_a_thread_1 is not user_a_thread_2
    assert user_b_thread_1 is not user_a_thread_2
    assert mock_pipeline.call_count == 3