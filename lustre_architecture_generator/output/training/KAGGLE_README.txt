LUSTRE MDT/OST LEARNING-TO-RANK DATA
====================================

Use case_id as the ranking group identifier.
Use relevance_grade as the primary ranking label.
Never use teacher_score or teacher_rank as input features.
Keep the provided split column; never split rows randomly.
Use model_feature_columns from the JSON manifest.
The deterministic hard filter must remain before ML.
