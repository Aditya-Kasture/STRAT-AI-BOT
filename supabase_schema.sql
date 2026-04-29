-- Run this in your Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- Required for items #3, #4, and #15 (persistent admin dashboard + multi-audit schema)

CREATE TABLE IF NOT EXISTS sessions (
  id                TEXT PRIMARY KEY,
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  completed_at      TIMESTAMPTZ,
  stage             TEXT DEFAULT 'intake',
  client_type       TEXT DEFAULT 'unknown',
  fit               TEXT DEFAULT 'unknown',
  contact_name      TEXT DEFAULT '',
  contact_email     TEXT DEFAULT '',
  company_name      TEXT DEFAULT '',
  api_cost_usd      FLOAT DEFAULT 0,
  api_calls         INTEGER DEFAULT 0,
  calendly_clicked  BOOLEAN DEFAULT FALSE,
  flags             JSONB DEFAULT '[]',
  snapshot_batch    INTEGER DEFAULT 0,
  snapshot_answers  JSONB DEFAULT '{}',
  deep_answers      JSONB DEFAULT '{}',
  qual_data         JSONB DEFAULT '{}',
  synthesis_text    TEXT DEFAULT '',
  deep_modules      JSONB DEFAULT '[]',
  deep_module_idx   INTEGER DEFAULT 0,
  metadata          JSONB DEFAULT '{}'
);

-- Indexes for common admin queries and future multi-audit dashboard (#15)
CREATE INDEX IF NOT EXISTS idx_sessions_email      ON sessions(contact_email);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_fit        ON sessions(fit);
CREATE INDEX IF NOT EXISTS idx_sessions_stage      ON sessions(stage);

-- Row-level security: allow server to read/write all rows
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

-- Policy: allow all operations (safe to re-run)
DROP POLICY IF EXISTS "service_full_access" ON sessions;
CREATE POLICY "service_full_access" ON sessions
  USING (true)
  WITH CHECK (true);
