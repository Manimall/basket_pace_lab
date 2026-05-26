from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    name: str = "basket_pace"
    user: str = "postgres"
    password: str = "postgres"

    @computed_field
    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @computed_field
    @property
    def sync_dsn(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    debug: bool = False
    log_level: str = "INFO"
    default_season: str = "2024-25"


class FeatureConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    score_roll_windows: tuple[int, ...] = (3, 5)
    pace_roll_windows: tuple[int, ...] = (3, 5, 10)
    ema_span: int = 5
    min_periods: int = 1
    days_rest_default: int = 7
    days_rest_clip_max: int = 21
    nba_playoff_start: str = "2026-04-19"

    # Current-season hard gate. Past seasons are treated as "information poison"
    # (different rosters, coaches, paces) — model trains only on matches at or
    # after this date. Bumped each new season.
    current_season_start: str = "2025-08-01"

    # Schedule-fatigue (V7) windowing knobs
    fatigue_short_window_days:      int = 4    # "3 in 4 nights" pattern
    fatigue_long_window_days:       int = 7    # weekly density context
    fatigue_high_density_threshold: int = 3    # games in short window → high-density flag


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    catboost_iterations: int = 800
    catboost_lr: float = 0.05
    catboost_depth: int = 5
    catboost_seed: int = 42
    test_frac: float = 0.20
    min_test_rows: int = 20


class EvaluationConfig(BaseSettings):
    """Backtester knobs: bet economics, probability sweep, bootstrap CI, guards.

    All values are env-overridable via the standard pydantic_settings mechanism
    (see model_config). Defaults match the V6 baseline documented in
    docs/backtester_evolution.md.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Bet economics
    odds: float = 1.90
    flat_stake: float = 1.0

    # Probability threshold sweep
    prob_thresholds: tuple[float, ...] = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60)

    # Bootstrap CI
    bootstrap_iters: int = 5000
    bootstrap_seed: int = 42
    bootstrap_ci_alpha: float = 0.05      # → 95% CI

    # Pipeline guards — minimum rows required to run a per-league pipeline
    min_train_rows: int = 100
    min_test_rows: int = 30


class CollectorConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Playwright request delays (seconds)
    delay_min: float = 2.0
    delay_max: float = 5.0
    season_pause_sec: float = 10.0
    # Collection limits
    max_pages: int = 50
    default_pages: int = 5
    # Sofascore defaults (NBA 25/26)
    default_tournament_id: int = 132
    default_season_id: int = 80229
    match_threshold: float = 0.40     # min name-similarity for Flashscore matching
    show_more_delay: float = 1.2      # seconds to wait after each "show more" click


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db: DatabaseSettings = DatabaseSettings()
    app: AppSettings = AppSettings()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    collector: CollectorConfig = CollectorConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


settings = Settings()
