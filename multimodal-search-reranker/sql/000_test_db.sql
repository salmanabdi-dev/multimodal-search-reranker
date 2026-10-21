-- Separate database for the integration tests, so they never touch real data.
-- (Its schema is created by the tests themselves via `init_db`.)
CREATE DATABASE mms_test;
