-- Mass forecast tables. {schema} is replaced by the job (public | europress).
-- Only CREATE IF NOT EXISTS: the job never drops or alters existing objects.

CREATE TABLE IF NOT EXISTS {schema}.mass_forecast_run (
    run_id          bigserial PRIMARY KEY,
    issued_at       timestamptz NOT NULL,          -- forecast origin: end of the last context hour
    created_at      timestamptz NOT NULL DEFAULT now(),
    model           text        NOT NULL,          -- experiment spec, e.g. timestral_cum@60+cal+fill+cycles
    model_revision  text,                          -- HF checkpoint / git revision
    context_hours   integer     NOT NULL,
    horizon_hours   integer     NOT NULL,
    n_devices       integer     NOT NULL,
    seconds         real,
    notes           text
);

CREATE TABLE IF NOT EXISTS {schema}.mass_forecast (
    device_id        text        NOT NULL,
    model            text        NOT NULL,
    issued_at        timestamptz NOT NULL,
    target_time      timestamptz NOT NULL,         -- end of the forecast hour (same convention as measured_at)
    lead_hours       smallint    NOT NULL,         -- target_time - issued_at
    mass_kg_q10      real,
    mass_kg_q50      real,
    mass_kg_q90      real,
    mass_kg_at_issue real,                         -- last observed mass in the context
    run_id           bigint      REFERENCES {schema}.mass_forecast_run (run_id),
    PRIMARY KEY (device_id, model, issued_at, target_time)
);

SELECT create_hypertable('{schema}.mass_forecast', 'issued_at', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS mass_forecast_target_idx ON {schema}.mass_forecast (device_id, target_time);

-- Latest forecast per device and model (one row per lead hour).
CREATE OR REPLACE VIEW {schema}.mass_forecast_latest AS
SELECT f.*
FROM {schema}.mass_forecast f
JOIN (SELECT device_id, model, max(issued_at) AS issued_at FROM {schema}.mass_forecast GROUP BY 1, 2) l
  USING (device_id, model, issued_at);
