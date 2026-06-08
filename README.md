# basket_pace_lab

> Лаборатория моделирования темпа и тотала в баскетболе через градиентный
> бустинг (CatBoost). Сквозной пайплайн: сбор данных → инженерия фичей →
> обучение → симуляция прибыли против закрывающей линии букмекера.

---

## 🧊 Статус

**V6 (frozen core) + per-league feature selection.** Ядро V6 заморожено как
инвариант: классификатор пробития линии, **ослеплённый к линии** (все
производные от линии фичи скрыты), сырые вероятности без калибровки, флэт-ставки,
хронологический per-league сплит, bootstrap CI95 на любой ROI.

Поверх ядра — **per-league feature selector**
([`src/evaluation/feature_selector.py`](src/evaluation/feature_selector.py)):
группы фичей `BASE / FATIGUE / TEAM_ADV / ADVANCED`, свой combo на каждую лигу,
строгий порог для ACB (0.60), бан убыточных лиг (CBA). Текущий набор —
**~58 фичей** (rolling Pace / ORtg / DRtg + усталость и плотность графика +
box-score advanced из Go-скаута `etl_scout`), а не 46.

**Вывод не изменился:** статзначимо положительного edge против закрывающей
линии не обнаружено. Подтверждено bootstrap CI95 (5000 итераций) на NBA +
контроль из 14 лиг, и повторно — на новых лигах (VTB / PLK / LKL, июнь 2026):
против реальной полноматчевой линии AUC ≤ 0.50, ROI отрицательный. Сами 1H/1Q
тоталы букмекерами по этим лигам не котируются (рынка нет).

- Краткий постмортем и pivot-план — [`docs/postmortem_v1_v6.md`](docs/postmortem_v1_v6.md)
- Детальный лог итераций V1→V6 — [`docs/backtester_evolution.md`](docs/backtester_evolution.md)

Дальнейшая работа — новые источники данных (late-scratch составы и травмы,
консенсус линий + Reverse Line Movement, назначения судей), а не новые
ML-приёмы на текущих фичах.

---

## 📊 Результаты V6 (probability threshold 0.50, ставки 1u flat, k=1.90)

| Пайплайн          | Bets | Winrate | ROI     | ROI CI95             |
|-------------------|------|---------|---------|----------------------|
| NBA               | 434  | 50.69%  | −3.69%  | [−12.44%; +5.07%]    |
| EuroLeague        | 80   | 46.25%  | −12.13% | [−33.50%; +9.25%]    |
| OTHER (контроль)  | 578  | 50.69%  | −3.69%  | [−11.57%; +4.20%]    |

На всех 18 ячейках (3 пайплайна × 6 порогов) нижняя граница CI95 не выходит
выше нуля — статзначимо положительного edge на текущих фичах не обнаружено.

---

## 🔧 Стек

| Слой              | Технология                          |
|-------------------|-------------------------------------|
| Язык              | Python 3.11+                        |
| База данных       | PostgreSQL 16                       |
| ORM               | SQLAlchemy 2.0 async + asyncpg      |
| Конфиг            | Pydantic v2 + pydantic-settings     |
| ML                | CatBoost, scikit-learn              |
| HTTP-парсинг      | httpx, curl_cffi                    |
| Браузер-парсинг   | Playwright (Flashscore odds)        |
| Тесты             | pytest                              |

---

## 🚀 Быстрый старт

### Docker Compose (рекомендуемый путь)

```bash
cp .env.example .env
docker compose up --build
```

Поднимаются два контейнера:

- **postgres** — PostgreSQL 16, том `postgres_data` для persistence
- **app** — Python-приложение, запускает `src/main.py` на старте
  (создаёт таблицы и проверяет коннект к БД)

Проверить, что таблицы созданы:

```bash
docker compose exec postgres psql -U postgres -d basket_pace -c "\dt"
```

Ожидаемые таблицы: `matches`, `quarter_stats`, `teams`.

Остановить:

```bash
docker compose down       # сохраняет volume
docker compose down -v    # сносит volume вместе с данными
```

### Локально (без Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium      # только если будете парсить Flashscore

# Postgres запустить отдельно, затем:
cp .env.example .env             # выставить DB_HOST=localhost
python -m src.main               # init таблиц + health-check
```

---

## 📐 Архитектура

```
┌──────────────────┐    ┌──────────────┐    ┌───────────────┐
│ data_collection/ │ →  │  PostgreSQL  │ →  │   features/   │
│ Flashscore       │    │   matches    │    │   rolling     │
│ Sofascore        │    │ quarter_stats│    │   EMA-5       │
│ ChinaNBL         │    │   teams      │    │   matchup     │
└──────────────────┘    └──────────────┘    └───────┬───────┘
                                                     │
                  ┌──────────────────────────────────┘
                  ▼
        ┌─────────────────┐    ┌──────────────────────┐
        │     models/     │ →  │     evaluation/      │
        │ CatBoost        │    │ симуляция + ROI CI95 │
        │ обучение, CV    │    │ per-league пайплайны │
        └─────────────────┘    └──────────────────────┘
```

### Модули `src/`

| Каталог            | Ответственность                                                   |
|--------------------|-------------------------------------------------------------------|
| `config/`          | `pydantic-settings`: БД, модель, фичи, бэктестер, парсеры         |
| `database/`        | SQLAlchemy ORM (`Match`, `QuarterStats`, `Team`), async CRUD      |
| `data_collection/` | Скрейперы: Flashscore (Playwright), Sofascore (httpx), ChinaNBL   |
| `features/`        | Rolling L3/L5/EMA-5, matchup-фичи (attack/defense, win-rate-diff) |
| `models/`          | Обучение CatBoost, per-league валидация                           |
| `evaluation/`      | Бэктестер: симуляция, bootstrap CI, per-league разбивка           |

---

## 🛠 Основные пайплайны

### 1. Сбор данных

```bash
# Flashscore Odds enricher по существующим match_id
python -m src.data_collection.flashscore.odds_enricher_direct --leagues NBA

# Сопоставление flashscore_id с DB-матчами из других источников
python -m src.data_collection.flashscore.id_matcher --leagues NBA BLeague
```

Полный CLI каждого скрейпера — в [`src/data_collection/`](src/data_collection/).

### 2. Обучение и валидация модели

```bash
python -m src.models.train_model          # MVP-регрессор q1_avg_pace
python -m src.models.validate_by_league   # CatBoost game_total + per-league MAE
```

### 3. Симуляция прибыли (бэктестер V6)

```bash
python -m src.evaluation.backtester
```

На выходе три таблицы (NBA / EuroLeague / OTHER) по 6 probability-порогам:
winrate, ROI и bootstrap CI95 на 5000 итераций.

---

## ⚙️ Конфигурация

Все параметры — в [`src/config/settings.py`](src/config/settings.py)
(pydantic-settings), с env-override через `.env`. Шаблон —
[`.env.example`](.env.example).

Основные секции:

| Секция         | Что внутри                                                    |
|----------------|---------------------------------------------------------------|
| `db.*`         | хост, порт, креды Postgres                                    |
| `model.*`      | гиперпараметры CatBoost, `test_frac`                          |
| `evaluation.*` | odds, prob_thresholds, bootstrap_iters, min_train/test_rows   |
| `features.*`   | окна rolling/EMA, days_rest                                   |
| `collector.*`  | задержки и лимиты Playwright-парсеров                         |

---

## 🧪 Тесты

```bash
pytest tests/
```

Текущее покрытие: **45 тестов** — фичи (rolling/EMA/matchup), парсеры
(team-name normalisation), pure-функции бэктестера (simulate / bootstrap CI
/ aggregate). Pure-логика симуляции в `src/evaluation/simulation.py`
покрыта полностью и работает без подключения к БД.

---

## 📚 Документация

- [`docs/postmortem_v1_v6.md`](docs/postmortem_v1_v6.md) — постмортем,
  статус заморозки, pivot-план на ABA + асимметричные данные
- [`docs/backtester_evolution.md`](docs/backtester_evolution.md) — детальный
  лог итераций V1→V6: гипотеза → что изменили → результат → что отбросили
  и почему

---

## 🗄 Дизайн БД: OT isolation

`QuarterStats.period_type` разделяет regulation-четверти (`QUARTER`) и
периоды овертайма (`OVERTIME`). Таблица `Match` хранит два варианта
итогового счёта:

- `*_score_final` — включая овертайм;
- `*_score_regulation` — только Q1-Q4.

ML-пайплайны **обязаны** использовать `*_score_regulation` как целевой
тотал и фильтровать строки с `q4_includes_ot_points = True`. Это
предотвращает утечку знания об овертайме в фичи Q1-Q4.
