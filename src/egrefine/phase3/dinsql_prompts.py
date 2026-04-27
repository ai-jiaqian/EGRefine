"""DIN-SQL prompt templates aligned with the official BIRD version.

Reference: MohammadrezaPourreza/Few-Shot-NL2SQL-with-Prompting (DIN-SQL_BIRD.py)

Key design choices from the official implementation:
- Schema Linking uses Chain-of-Thought ("Let's think step by step")
- Classification uses explicit decision rules (EASY/NON-NESTED/NESTED)
- SQL Generation uses difficulty-specific reasoning strategies
- Self-correction uses a rule-based zero-shot prompt with explicit checks
- BIRD hints/evidence are included in all prompts
"""

# ------------------------------------------------------------------ #
# Module 1: Schema Linking (with CoT)
# ------------------------------------------------------------------ #

SCHEMA_LINKING_INSTRUCTION = """\
# Find the schema_links for generating SQL queries for each question \
based on the database schema and Foreign keys.

# Use Chain-of-Thought reasoning to identify which tables and columns \
are referenced by each phrase in the question.

# Output the schema_links as a list of table.column references."""

SCHEMA_LINKING_EXEMPLARS = [
    {
        "schema": (
            "CREATE TABLE lists (\n"
            "  user_id INTEGER,\n"
            "  list_id INTEGER PRIMARY KEY,\n"
            "  list_title TEXT,\n"
            "  list_creation_date_utc TEXT,\n"
            "  list_followers INTEGER,\n"
            "  list_url TEXT,\n"
            "  list_description TEXT\n"
            ");\n\n"
            "CREATE TABLE movies (\n"
            "  movie_id INTEGER PRIMARY KEY,\n"
            "  movie_title TEXT,\n"
            "  movie_release_year INTEGER,\n"
            "  movie_url TEXT,\n"
            "  movie_title_language TEXT,\n"
            "  movie_popularity INTEGER,\n"
            "  movie_image_url TEXT,\n"
            "  director_id TEXT,\n"
            "  director_name TEXT,\n"
            "  director_url TEXT\n"
            ");\n\n"
            "CREATE TABLE ratings (\n"
            "  movie_id INTEGER,\n"
            "  rating_id INTEGER,\n"
            "  rating_url TEXT,\n"
            "  rating_score INTEGER,\n"
            "  rating_timestamp_utc TEXT,\n"
            "  critic TEXT,\n"
            "  critic_likes INTEGER,\n"
            "  critic_comments INTEGER,\n"
            "  user_id INTEGER,\n"
            "  user_trialist INTEGER,\n"
            "  user_subscriber INTEGER,\n"
            "  user_eligible_for_trial INTEGER,\n"
            "  user_has_payment_method INTEGER,\n"
            "  FOREIGN KEY (movie_id) REFERENCES movies(movie_id),\n"
            "  FOREIGN KEY (user_id) REFERENCES lists(user_id)\n"
            ");\n\n"
            "Foreign_keys = [ratings.movie_id = movies.movie_id, "
            "ratings.user_id = lists.user_id]"
        ),
        "question": "What is the name of the movie that was rated by the highest number of users who are trialists?",
        "hint": "number of users who are trialists refers to user_trialist = 1",
        "output": (
            "Let's think step by step. "
            "In the question \"What is the name of the movie\", the name of the movie refers to "
            "movie_title in the movies table. "
            "\"was rated\" refers to the ratings table. "
            "\"the highest number of users who are trialists\" refers to MAX(COUNT(user_id)) "
            "with user_trialist = 1 in the ratings table. "
            "Based on the hint, user_trialist = 1 is in the ratings table.\n"
            "Schema_links: [movies.movie_title, ratings.movie_id, movies.movie_id, "
            "ratings.user_id, ratings.user_trialist]"
        ),
    },
    {
        "schema": (
            "CREATE TABLE lists (\n"
            "  user_id INTEGER,\n"
            "  list_id INTEGER PRIMARY KEY,\n"
            "  list_title TEXT,\n"
            "  list_creation_date_utc TEXT,\n"
            "  list_followers INTEGER,\n"
            "  list_url TEXT,\n"
            "  list_description TEXT\n"
            ");\n\n"
            "CREATE TABLE movies (\n"
            "  movie_id INTEGER PRIMARY KEY,\n"
            "  movie_title TEXT,\n"
            "  movie_release_year INTEGER,\n"
            "  movie_url TEXT,\n"
            "  movie_title_language TEXT,\n"
            "  movie_popularity INTEGER,\n"
            "  movie_image_url TEXT,\n"
            "  director_id TEXT,\n"
            "  director_name TEXT,\n"
            "  director_url TEXT\n"
            ");\n\n"
            "CREATE TABLE ratings (\n"
            "  movie_id INTEGER,\n"
            "  rating_id INTEGER,\n"
            "  rating_url TEXT,\n"
            "  rating_score INTEGER,\n"
            "  rating_timestamp_utc TEXT,\n"
            "  critic TEXT,\n"
            "  critic_likes INTEGER,\n"
            "  critic_comments INTEGER,\n"
            "  user_id INTEGER,\n"
            "  user_trialist INTEGER,\n"
            "  user_subscriber INTEGER,\n"
            "  user_eligible_for_trial INTEGER,\n"
            "  user_has_payment_method INTEGER,\n"
            "  FOREIGN KEY (movie_id) REFERENCES movies(movie_id),\n"
            "  FOREIGN KEY (user_id) REFERENCES lists(user_id)\n"
            ");\n\n"
            "Foreign_keys = [ratings.movie_id = movies.movie_id, "
            "ratings.user_id = lists.user_id]"
        ),
        "question": "How many users who were a subscriber when they rated the movie Mo rated the movie Mo with a rating score of 5?",
        "hint": "user_subscriber = 1 when the user was a subscriber",
        "output": (
            "Let's think step by step. "
            "In the question \"How many users\", the count refers to COUNT(user_id) in the "
            "ratings table. "
            "\"who were a subscriber\" refers to user_subscriber = 1 in the ratings table, "
            "confirmed by the hint. "
            "\"when they rated the movie\" refers to the ratings table and joining with movies. "
            "\"the movie Mo\" refers to movie_title = 'Mo' in the movies table. "
            "\"with a rating score of 5\" refers to rating_score = 5 in the ratings table.\n"
            "Schema_links: [ratings.user_id, ratings.user_subscriber, "
            "ratings.rating_score, movies.movie_title, ratings.movie_id, movies.movie_id]"
        ),
    },
]

# ------------------------------------------------------------------ #
# Module 2: Query Classification (with explicit rules)
# ------------------------------------------------------------------ #

CLASSIFICATION_INSTRUCTION = """\
# For the given question, classify it as EASY, NON-NESTED, or NESTED \
based on the schema links and nested query requirements.

# Use the following rules:
# if the question can be solved with a nested query, classify as "NESTED"
# elif the question requires joining multiple tables, classify as "NON-NESTED"
# elif the question can be answered from a single table without JOIN, classify as "EASY"
"""

CLASSIFICATION_EXEMPLARS = [
    {
        "question": "What is the name of the movie that was rated by the highest number of users who are trialists?",
        "hint": "number of users who are trialists refers to user_trialist = 1",
        "linking": (
            "Schema_links: [movies.movie_title, ratings.movie_id, movies.movie_id, "
            "ratings.user_id, ratings.user_trialist]"
        ),
        "label": (
            "Let's think step by step. "
            "The question asks for the movie name with the highest count of trialist users. "
            "This requires joining movies and ratings tables, then grouping and ordering. "
            "It needs a subquery or ORDER BY ... LIMIT 1 pattern. "
            "Multiple tables are needed → NON-NESTED or NESTED. "
            "Since we can solve this without a nested subquery (using GROUP BY + ORDER BY + LIMIT), "
            "classify as \"NON-NESTED\""
        ),
    },
    {
        "question": "How many users who were a subscriber rated the movie Mo with a score of 5?",
        "hint": "user_subscriber = 1 means subscriber",
        "linking": (
            "Schema_links: [ratings.user_id, ratings.user_subscriber, "
            "ratings.rating_score, movies.movie_title, ratings.movie_id, movies.movie_id]"
        ),
        "label": (
            "Let's think step by step. "
            "We need to count users from ratings joined with movies. "
            "Filters: user_subscriber = 1, movie_title = 'Mo', rating_score = 5. "
            "This requires a JOIN but no nested subquery. "
            "Classify as \"NON-NESTED\""
        ),
    },
    {
        "question": "How many lists does the user who created the list 'Sound4Film' have?",
        "hint": "",
        "linking": "Schema_links: [lists.user_id, lists.list_title, lists.list_id]",
        "label": (
            "Let's think step by step. "
            "We need to find the user who created 'Sound4Film', then count their lists. "
            "This requires a subquery: first find user_id WHERE list_title = 'Sound4Film', "
            "then COUNT lists WHERE user_id = that result. "
            "Classify as \"NESTED\""
        ),
    },
    {
        "question": "How many movies were released in 2020?",
        "hint": "",
        "linking": "Schema_links: [movies.movie_id, movies.movie_release_year]",
        "label": (
            "Let's think step by step. "
            "We need COUNT from movies where movie_release_year = 2020. "
            "Single table, no JOIN needed. "
            "Classify as \"EASY\""
        ),
    },
    {
        "question": "What is the URL of the list created by user 85981819?",
        "hint": "",
        "linking": "Schema_links: [lists.list_url, lists.user_id]",
        "label": (
            "Let's think step by step. "
            "Select list_url from lists where user_id = 85981819. "
            "Single table, no JOIN needed. "
            "Classify as \"EASY\""
        ),
    },
    {
        "question": "Which movies have a higher popularity than the average popularity of all movies?",
        "hint": "",
        "linking": "Schema_links: [movies.movie_title, movies.movie_popularity]",
        "label": (
            "Let's think step by step. "
            "We need movies where popularity > average popularity. "
            "The average requires a subquery: SELECT AVG(movie_popularity) FROM movies. "
            "Classify as \"NESTED\""
        ),
    },
]

# ------------------------------------------------------------------ #
# Module 3: SQL Generation (difficulty-specific with reasoning)
# ------------------------------------------------------------------ #

# ----- EASY: direct Q → Schema_links → SQL -----

GENERATION_INSTRUCTION_EASY = """\
# Use the the schema links and hints to generate the correct sqlite SQL query \
for the given question.

# Only output the SQL query, no explanation."""

GENERATION_EXEMPLARS_EASY = [
    {
        "question": "How many movies were released in 2020?",
        "hint": "",
        "linking": "Schema_links: [movies.movie_id, movies.movie_release_year]",
        "sql": "SELECT COUNT(movie_id) FROM movies WHERE movie_release_year = 2020",
    },
    {
        "question": "What is the URL of the list created by user 85981819?",
        "hint": "",
        "linking": "Schema_links: [lists.list_url, lists.user_id]",
        "sql": "SELECT list_url FROM lists WHERE user_id = 85981819",
    },
    {
        "question": "How many movie lists were created after 2010?",
        "hint": "created after 2010 refers to list_creation_date_utc > '2010-12-31'",
        "linking": "Schema_links: [lists.list_id, lists.list_creation_date_utc]",
        "sql": (
            "SELECT COUNT(list_id) FROM lists "
            "WHERE list_creation_date_utc > '2010-12-31'"
        ),
    },
    {
        "question": "What is the name of the director of the most popular movie?",
        "hint": "most popular movie refers to MAX(movie_popularity)",
        "linking": "Schema_links: [movies.director_name, movies.movie_popularity]",
        "sql": (
            "SELECT director_name FROM movies "
            "ORDER BY movie_popularity DESC LIMIT 1"
        ),
    },
    {
        "question": "How many lists have more than 100 followers?",
        "hint": "",
        "linking": "Schema_links: [lists.list_id, lists.list_followers]",
        "sql": "SELECT COUNT(list_id) FROM lists WHERE list_followers > 100",
    },
    {
        "question": "What is the average rating score for all movies?",
        "hint": "",
        "linking": "Schema_links: [ratings.rating_score]",
        "sql": "SELECT AVG(rating_score) FROM ratings",
    },
]

# ----- NON-NESTED: with intermediate representation -----

GENERATION_INSTRUCTION_NON_NESTED = """\
# Use the the schema links and hints to generate the correct sqlite SQL query \
for the given question.

# First think step by step about which tables need to be joined and the join \
conditions from the foreign keys, then write the SQL.

# Only output the final SQL query after the reasoning."""

GENERATION_EXEMPLARS_NON_NESTED = [
    {
        "question": "What is the name of the movie that was rated by the highest number of users who are trialists?",
        "hint": "number of users who are trialists refers to user_trialist = 1",
        "linking": (
            "Schema_links: [movies.movie_title, ratings.movie_id, movies.movie_id, "
            "ratings.user_id, ratings.user_trialist]"
        ),
        "sql": (
            "Let's think step by step. For creating the SQL for this question, "
            "we need to join these tables = [movies, ratings]. "
            "Join condition: movies.movie_id = ratings.movie_id. "
            "Filter: ratings.user_trialist = 1. "
            "Aggregate: GROUP BY movie_title, ORDER BY COUNT(user_id) DESC LIMIT 1.\n"
            "SQL: SELECT T1.movie_title FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T2.user_trialist = 1 "
            "GROUP BY T1.movie_title "
            "ORDER BY COUNT(T2.user_id) DESC LIMIT 1"
        ),
    },
    {
        "question": "How many users who were a subscriber rated the movie Mo with a score of 5?",
        "hint": "user_subscriber = 1 means subscriber",
        "linking": (
            "Schema_links: [ratings.user_id, ratings.user_subscriber, "
            "ratings.rating_score, movies.movie_title, ratings.movie_id, movies.movie_id]"
        ),
        "sql": (
            "Let's think step by step. For creating the SQL for this question, "
            "we need to join these tables = [movies, ratings]. "
            "Join condition: movies.movie_id = ratings.movie_id. "
            "Filters: user_subscriber = 1, movie_title = 'Mo', rating_score = 5.\n"
            "SQL: SELECT COUNT(T2.user_id) FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T2.user_subscriber = 1 AND T1.movie_title = 'Mo' "
            "AND T2.rating_score = 5"
        ),
    },
    {
        "question": "What is the average rating score for movies directed by Steven Spielberg?",
        "hint": "",
        "linking": (
            "Schema_links: [ratings.rating_score, movies.director_name, "
            "ratings.movie_id, movies.movie_id]"
        ),
        "sql": (
            "Let's think step by step. For creating the SQL for this question, "
            "we need to join these tables = [movies, ratings]. "
            "Join condition: movies.movie_id = ratings.movie_id. "
            "Filter: director_name = 'Steven Spielberg'. "
            "Aggregate: AVG(rating_score).\n"
            "SQL: SELECT AVG(T2.rating_score) FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T1.director_name = 'Steven Spielberg'"
        ),
    },
    {
        "question": "List the titles of movies rated by subscribers, ordered by rating score.",
        "hint": "subscriber refers to user_subscriber = 1",
        "linking": (
            "Schema_links: [movies.movie_title, ratings.rating_score, "
            "ratings.user_subscriber, ratings.movie_id, movies.movie_id]"
        ),
        "sql": (
            "Let's think step by step. For creating the SQL for this question, "
            "we need to join these tables = [movies, ratings]. "
            "Join condition: movies.movie_id = ratings.movie_id. "
            "Filter: user_subscriber = 1. "
            "Order: rating_score DESC.\n"
            "SQL: SELECT T1.movie_title FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T2.user_subscriber = 1 "
            "ORDER BY T2.rating_score DESC"
        ),
    },
]

# ----- NESTED: with sub-question decomposition -----

GENERATION_INSTRUCTION_NESTED = """\
# Use the the schema links and hints to generate the correct sqlite SQL query \
for the given question.

# Decompose the question into sub-questions if needed, solve each sub-question, \
and then compose the final SQL.

# Only output the final SQL query after the reasoning."""

GENERATION_EXEMPLARS_NESTED = [
    {
        "question": "How many lists does the user who created the list 'Sound4Film' have?",
        "hint": "",
        "linking": "Schema_links: [lists.user_id, lists.list_title, lists.list_id]",
        "sql": (
            "Let's think step by step. "
            "This question can be solved by first finding the user who created 'Sound4Film', "
            "then counting their lists.\n"
            "Sub-question: What is the user_id of the user who created 'Sound4Film'? "
            "SQL: SELECT user_id FROM lists WHERE list_title = 'Sound4Film'\n"
            "Main query: Count lists for that user.\n"
            "SQL: SELECT COUNT(list_id) FROM lists "
            "WHERE user_id = (SELECT user_id FROM lists WHERE list_title = 'Sound4Film')"
        ),
    },
    {
        "question": "Which movies have a higher popularity than the average popularity of all movies?",
        "hint": "",
        "linking": "Schema_links: [movies.movie_title, movies.movie_popularity]",
        "sql": (
            "Let's think step by step. "
            "This question can be solved by first computing the average popularity, "
            "then filtering movies above it.\n"
            "Sub-question: What is the average movie popularity? "
            "SQL: SELECT AVG(movie_popularity) FROM movies\n"
            "Main query: Select movies with popularity above average.\n"
            "SQL: SELECT movie_title FROM movies "
            "WHERE movie_popularity > (SELECT AVG(movie_popularity) FROM movies)"
        ),
    },
    {
        "question": "What are the titles of movies that have been rated by more users than the movie 'Inception'?",
        "hint": "",
        "linking": (
            "Schema_links: [movies.movie_title, ratings.movie_id, movies.movie_id, "
            "ratings.user_id]"
        ),
        "sql": (
            "Let's think step by step. "
            "This question can be solved by first finding how many users rated 'Inception', "
            "then finding movies rated by more users.\n"
            "Sub-question: How many users rated 'Inception'? "
            "SQL: SELECT COUNT(T2.user_id) FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T1.movie_title = 'Inception'\n"
            "Main query: Find movies with more raters.\n"
            "SQL: SELECT T1.movie_title FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "GROUP BY T1.movie_title "
            "HAVING COUNT(T2.user_id) > "
            "(SELECT COUNT(T2.user_id) FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T1.movie_title = 'Inception')"
        ),
    },
]

# ------------------------------------------------------------------ #
# Module 4: Self-Correction (rule-based, zero-shot style)
# ------------------------------------------------------------------ #

SELF_CORRECTION_INSTRUCTION = """\
#### For the given question, use the provided database schema and hint \
to fix the given SQLite SQL QUERY for any issues. If there are any \
problems, fix them. If there are no issues, return the SQL QUERY as is.

#### Use the following rules to check the SQL:
1. Check that all column names exist in the referenced tables
2. Check that JOIN conditions use the correct foreign keys
3. Check that GROUP BY includes all non-aggregated SELECT columns
4. Check that WHERE/HAVING logic matches the question intent
5. Use DISTINCT when the question implies unique results
6. Use DESC for "highest/most/largest" and ASC for "lowest/least/smallest"
7. Ensure correct use of aggregation functions (COUNT, SUM, AVG, MAX, MIN)
8. Use CAST when comparing different data types
9. Check column data type matches the comparison values

#### Return only the fixed SQL query, no explanation."""

# Self-correction uses zero-shot (official Spider version) or 1 exemplar (BIRD version).
# We include 1 exemplar to match the BIRD version.
SELF_CORRECTION_EXEMPLARS = [
    {
        "question": "What is the name of the highest rated movie directed by Steven Spielberg?",
        "hint": "highest rated refers to MAX(rating_score)",
        "schema": (
            "CREATE TABLE movies (movie_id INTEGER PRIMARY KEY, movie_title TEXT, "
            "director_name TEXT);\n"
            "CREATE TABLE ratings (movie_id INTEGER, rating_score INTEGER, "
            "FOREIGN KEY (movie_id) REFERENCES movies(movie_id));"
        ),
        "sql": (
            "SELECT T1.movie_title FROM movies AS T1 "
            "JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T1.director_name = 'Steven Spielberg' "
            "GROUP BY T1.movie_title ORDER BY rating_score LIMIT 1"
        ),
        "corrected": (
            "SELECT T1.movie_title FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
            "WHERE T1.director_name = 'Steven Spielberg' "
            "GROUP BY T1.movie_title "
            "ORDER BY AVG(T2.rating_score) DESC LIMIT 1"
        ),
    },
]
