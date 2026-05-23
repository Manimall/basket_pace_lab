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


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    catboost_iterations: int = 800
    catboost_lr: float = 0.05
    catboost_depth: int = 5
    catboost_seed: int = 42
    test_frac: float = 0.20
    min_test_rows: int = 20


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db: DatabaseSettings = DatabaseSettings()
    app: AppSettings = AppSettings()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()


settings = Settings()
