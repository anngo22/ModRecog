# ModRecog: классификация типов модуляции сигналов

Горбунова Анна Олеговна

В качестве DVC-хранилища используется локальное хранилище (`data-store` для датасета, `model-store` для моделей).
Из-за этого данные и модели недоступны без локального доступа к хранилищам.
Однако все команды можно воспроизвести, загрузив датасет с [Kaggle](https://www.kaggle.com/datasets/pinxau1000/radioml2018/data) и обучив модель.

---

## Постановка задачи

Разработать классификатор, который по сырым I/Q-отсчётам радиосигнала определяет тип его модуляции среди 24 возможных классов.
Задача актуальна для систем радиомониторинга, когнитивного радио и автоматического распознавания сигналов (AMR/AMC).
За основу взята архитектура и датасет из [статьи O'Shea & Hoydis (2017)](https://arxiv.org/pdf/1712.04578).

---

## Формат входных и выходных данных

**Вход:** тензор размером `(batch_size, 2, 1024)`, где:

- `2` — каналы I (синфазная) и Q (квадратурная) составляющие сигнала,
- `1024` — длина окна сэмплирования.

**Пример одного фрейма** (форма `[2, 1024]`, значения float32):

```json
[
  [0.0231, -0.0114, 0.0453, -0.0317, 0.0189, "... (1024 значения) ..."],
  [-0.0127, 0.0342, -0.0561, 0.0208, -0.0093, "... (1024 значения) ..."]
]
```

**Выход:** вектор вероятностей размером `(batch_size, 24)` — принадлежность каждого фрейма к одному из 24 классов модуляции.

**Пример ответа инференса:**

```
[0] 32PSK (id=0, conf=0.9325)
```

Классы (24 штуки): `32PSK`, `16APSK`, `32QAM`, `FM`, `GMSK`, `32APSK`, `OQPSK`, `8ASK`, `BPSK`, `8PSK`,
`AM-SSB-SC`, `4ASK`, `16PSK`, `64APSK`, `128QAM`, `128APSK`, `AM-DSB-SC`, `AM-SSB-WC`, `64QAM`, `QPSK`,
`256QAM`, `AM-DSB-WC`, `OOK`, `16QAM`.

---

## Метрики

Используются пять метрик, которые также логируются в MLflow:

| Метрика             | Бейзлайн (XGBoost) | ResNet1D | Логируется как                       |
| ------------------- | ------------------ | -------- | ------------------------------------ |
| **Accuracy**        | ~40%               | ~56%     | `train/acc` · `val/acc` · `test/acc` |
| **Macro F1-score**  | ~35%               | ~52%     | `val/f1` · `test/f1`                 |
| **Top-3 Accuracy**  | ~65%               | ~78%     | `val/top3_acc` · `test/top3_acc`     |
| **Macro Precision** | ~35%               | ~53%     | `val/precision` · `test/precision`   |
| **Macro Recall**    | ~35%               | ~52%     | `val/recall` · `test/recall`         |

Ожидаемые значения для ResNet1D основаны на результатах из [O'Shea & Hoydis (2017)](https://arxiv.org/pdf/1712.04578)
и усреднены по всем уровням SNR (-20 ... 30 дБ). При SNR < 0 дБ сигнал сильно зашумлён и точность близка
к случайному угадыванию (1/24 ≈ 4%), при SNR ≥ 10 дБ — превышает 90%.
Для XGBoost полный per-class отчёт (`precision`, `recall`, `f1` на каждый из 24 классов) выводится в консоль скриптом `baseline_xgb.py`.

---

## Валидация и тест

Стратифицированное разбиение по классу модуляции: **train 80%** · **val 10%** · **test 10%**.
Стратификация обеспечивает одинаковое распределение классов во всех сплитах.
Для воспроизводимости фиксируется `seed = 42`.

---

## Данные

Датасет: [RadioML 2018.01A](https://www.kaggle.com/datasets/pinxau1000/radioml2018/data). Публичный датасет синтетических и реальных радиосигналов.

- **Объём:** 2 555 904 образца (24 класса × 26 уровней SNR × 4 096 фреймов)
- **Размер на диске:** ~19.9 ГБ
- **SNR:** от −20 до 30 дБ с шагом 2 дБ (26 уровней)
- **Каждый фрейм:** 1024 комплексных I/Q-отсчёта (-> `(1024, 2)` float32)
- **Баланс классов:** строго сбалансирован: по 4 096 фреймов на каждую пару (класс, SNR).

Для train, val и test данные делятся в соотношении 80/10/10 с сохранением баланса классов.
При низком SNR (< 0 дБ) сигнал сильно зашумлён, поэтому классы становятся неразличимы, и модель работает почти как случайная.

---

## Моделирование

### Бейзлайн

Классический подход на основе статистических признаков из [той же статьи](https://arxiv.org/pdf/1712.04578).

Пайплайн: загрузка HDF5, фильтрация по SNR -> извлечение 26 статистических признаков (по 13 на каждый из каналов I и Q:
среднее, std, итд.)
-> нормализация `StandardScaler` -> `XGBClassifier` (300 деревьев, `max_depth=6`) -> `argmax` -> имя класса.

### Основная модель

**ResNet1D** из [O'Shea & Hoydis (2017)](https://arxiv.org/pdf/1712.04578).

Пайплайн: загрузка HDF5 с транспонированием `(1024, 2)` -> `(2, 1024)` -> 6 residual stacks (фильтры `[32, 32, 64, 64, 128, 128]`,
ядро = 3, skip-connection через `Conv1d(1×1)`, активация SELU, AlphaDropout p=0.1) -> GlobalAveragePooling1D ->
Linear(128 -> 24) -> `softmax` -> `argmax` -> имя класса. Параметров: ~255 тыс.

Обучение: Adam (lr = 1e-3), ReduceLROnPlateau (patience=5, factor=0.5), CrossEntropy loss,
EarlyStopping по val_loss (patience=10), ModelCheckpoint (top-1 по val/f1).

---

## Внедрение

Модель экспортируется в ONNX и разворачивается через **Triton Inference Server**.
Сервис принимает батч I/Q-фреймов `(N, 2, 1024)` и возвращает логиты `(N, 24)`.
Постобработка на стороне клиента: `softmax(logits)` -> `argmax` -> имя класса.

| Формат                       | Путь                                  | Назначение                     |
| ---------------------------- | ------------------------------------- | ------------------------------ |
| PyTorch Lightning checkpoint | `models/dvc/best-*.ckpt`              | дообучение, локальный инференс |
| ONNX                         | `models/triton/modrecog/1/model.onnx` | кросс-платформенный инференс   |

Размер модели: ~255 тыс. параметров (~1 МБ ONNX-файл).

---

## Структура проекта

```
ModRecog/
├── configs/                    # Hydra-конфиги
│   ├── data/radioml2018.yaml
│   ├── logging/mlflow.yaml
│   ├── model/resnet.yaml
│   ├── training/default.yaml
│   └── config.yaml             # точка входа с defaults
├── data/
│   └── raw.dvc                 # DVC-указатель на датасет
├── models/
│   ├── dvc.dvc                 # DVC-указатель на чекпоинты
│   └── triton/modrecog/
│       ├── 1/                  # model.onnx (не в git, создаётся export-onnx)
│       └── config.pbtxt        # конфигурация Triton
├── modrecog/
│   ├── data.py                 # RadioMLDataModule
│   ├── export.py               # ONNX экспорт
│   ├── infer.py                # инференс из чекпоинта
│   ├── infer_triton.py         # HTTP-клиент Triton
│   ├── model.py                # ResNet1D + LightningModule
│   ├── train.py                # обучение
│   └── utils.py                # утилиты, MODULATION_CLASSES
├── scripts/
│   ├── baseline_xgb.py         # XGBoost-бейзлайн
│   └── make_samples_json.py    # генерация тестовых данных для инференса
├── tests/
│   └── test_model.py
├── commands.py                 # CLI-точка входа (fire + hydra)
├── docker-compose.yaml         # MLflow + Triton
├── pyproject.toml
└── README.md
```

---

## Setup

Разработка велась на Windows 10, Python 3.11. Для установки зависимостей используется poetry.

Начало работы:

```bash
git clone https://github.com/anngo22/ModRecog
cd ModRecog
```

Создать и активировать виртуальное окружение:

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

> Ubuntu/Debian. Если `python -m .venv` завершается с ошибкой `ensurepip is not available`, запустите:
>
> ```powershell
> sudo apt install python3.12-venv
> ```

Установить зависимости:

```bash
pip install poetry
poetry install
```

Установить и проверить pre-commit хуки:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

---

## Train

Сначала запускается:

```bash
docker compose up mlflow -d
# или:
mlflow server --host 127.0.0.1 --port 8080
```

Предполагается, что MLflow server уже запущен.

Затем:

```bash
poetry run python commands.py download-data
```

Запущенный процесс скачает данные с Kaggle, если не сможет их получить из DVC-хранилища (`dvc pull --remote data-store`).
Требуется `~/.kaggle/kaggle.json` или переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`.

Запуск тренировки (пример на CPU для `SNR >= 0`):

```bash
# Быстрый запуск на CPU: 50 тыс. сэмплов, SNR >= 0 дБ, 30 эпох
poetry run python commands.py train \
    data.max_samples=50000 \
    data.snr_min=0 \
    training.epochs=30 \
    data.num_workers=0

# Полное обучение на GPU:
poetry run python commands.py train
```

После завершения обучения модель автоматически добавляется в DVC (`dvc add` + `dvc push --remote model-store`).
Все метрики и гиперпараметры логируются в MLflow UI: <http://127.0.0.1:8080>.
Кривые loss / accuracy / F1 сохраняются в `plots/training_curves.png`.

Для сравнения с бейзлайном запускается с теми же параметрами, что и основная модель:

```bash
poetry run python scripts/baseline_xgb.py --snr_min 0 --max_samples 50000
```

---

## Production preparation

**ONNX.** Экспорт чекпоинта в ONNX-формат для Triton:

```bash
poetry run export-onnx --checkpoint models/dvc/last.ckpt
```

Файл сохраняется в `models/triton/modrecog/1/model.onnx`.
Конфигурация Triton находится в `models/triton/modrecog/config.pbtxt`.

**Triton Inference Server.** Запуск локального сервера:

```bash
docker compose up triton -d
```

Сервер поднимается на портах `8000` (HTTP), `8001` (gRPC), `8002` (метрики).

---

## Infer

**Входные данные.** Инференс принимает JSON-файл (массив фреймов, размерности `(2, 1024)`).
Для генерации тестового файла из датасета введите:

```bash
# 3 сэмпла с SNR = 10 дБ
poetry run python scripts/make_samples_json.py --n 3 --snr 10
```

Результатом будут истинные классы для последующей проверки:

```
Wrote 3 sample(s) to samples.json
  [0] true class: BPSK  (SNR +10 dB)
  [1] true class: QPSK  (SNR +10 dB)
  [2] true class: 16QAM (SNR +10 dB)
```

**Запуск инференса из чекпоинта:**

```bash
poetry run python commands.py infer samples.json
# или:
poetry run python commands.py infer samples.json --checkpoint models/dvc/last.ckpt
```

**Запуск инференса из Triton Inference Server:**

```bash
poetry run python commands.py infer-triton samples.json
```

Пример вывода:

```
[0] BPSK  (id=8,  conf=0.9821)
[1] QPSK  (id=19, conf=0.9634)
[2] 16QAM (id=23, conf=0.8912)
```

---

## Ссылки

- T. O'Shea & J. Hoydis, «An Introduction to Deep Learning for the Physical Layer», 2017:
  <https://arxiv.org/pdf/1712.04578>
- RadioML 2018.01A на Kaggle:
  <https://www.kaggle.com/datasets/pinxau1000/radioml2018/data>
