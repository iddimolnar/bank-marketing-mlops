# Tanulói napló — Bank Marketing MLOps

Ez a fájl lépésről lépésre elmagyarázza, **mikor mit csinálunk és miért**. Nem a kód
részleteiről szól, hanem a gondolkodásról. Fázisonként bővül, ahogy haladunk.

---

## Nagy kép

A féléves feladat: egy **teljes MLOps rendszer** a Bank Marketing adathalmazra. Nem a
modell pontossága a lényeg, hanem hogy **production-ready** módon épüljön fel:
verziózva, tesztelve, konténerizálva, monitorozva, egyetlen paranccsal indíthatóan.

Fázisokban építjük, mindig úgy, hogy a végén **működő és bekommitolt** állapot legyen:

| Fázis | Mit ad hozzá |
|------|--------------|
| 0 | Projekt-váz, eszközök (git, tesztelő, linter, CI) |
| 1 | Konfiguráció, naplózás, adat-sémák |
| 2 | Adatbetöltés, előfeldolgozás, modell |
| 3 | Tanítás + MLflow követés + model registry + promóciós kapu |
| 4 | Predikció + kérés-naplózás |
| 5 | REST API (FastAPI) |
| 6 | Docker + `docker compose up` |
| 7 | Evidently drift riport + Streamlit dashboard |
| 8 | CI szigorítás (coverage, típus) |
| 9 | Airflow (ütemezett újratanítás) |
| 10 | Dokumentáció, screenshotok, beadás |

---

## Fogalmak röviden (amikre hivatkozunk)

- **venv**: izolált Python-környezet, hogy a projekt csomagjai ne keveredjenek a gépeddel.
- **flake8**: PEP8 (kódstílus) ellenőrző. Szabály: max 1% hiba.
- **pyright**: típusellenőrző. Szabály: 0 hiba az `src/`-ben.
- **pytest + coverage**: tesztek futtatása és lefedettség mérése. Szabály: ≥95%.
- **pydantic**: adat-validáció típusok alapján; a "rossz adat" már a határon elbukik.
- **pre-commit / GitHub Actions**: automatikusan futtatják a fentieket commit/push előtt
  és a felhőben, hogy ne kerülhessen be hibás kód.

---

## Fázis 0 — Projekt-váz és eszközök

**Cél:** legyen egy üres, de "felnőtt" projekt, amiben már minden minőségi kapu működik.

Mit hoztunk létre és miért:

- **git repo** (`git init`) + `.gitignore`: a kódot verziózzuk; a `.gitignore` kizárja azt,
  amit soha nem szabad bekommitolni (venv, titkok, generált adat, modell-fájlok).
- **`src/` elrendezés** (`src/bank_marketing/`): a csomag csak telepítés után importálható
  — ez megakadályozza a "nálam a gyökérből működött" típusú hibákat.
- **`pyproject.toml`**: egységes projekt-leírás (függőségek + eszköz-beállítások egy helyen).
- **`requirements.txt` pontos verziókkal**: reprodukálható környezet (ugyanaz fut nálad és a CI-ben).
- **`.flake8`, pyright, ruff beállítás**: a kódstílus és a típusok szabályai.
- **`.pre-commit-config.yaml`**: minden commitnál lefut a linter/formázó; push előtt a tesztek.
- **GitHub Actions (`ci.yml`)**: a felhőben is lefut lint + típus + teszt + coverage.
- **`tiny_bank.csv` fixture**: egy kicsi, mindkét osztályt tartalmazó minta, hogy a tesztek
  gyorsan fussanak és ne kelljen hozzá a nagy adat.

**Tanulság:** előbb a "korlát", utána a kód. Így minden új sor azonnal ellenőrzött.

---

## Fázis 1 — Konfiguráció, naplózás, sémák

**Cél:** a rendszer "idegrendszere" — hogyan kap beállításokat, hogyan naplóz, és hogyan
ismeri fel a jó/rossz bemenetet.

Három modul készült, mindegyik egyetlen felelősséggel:

1. **`config.py` — beállítások (pydantic-settings).**
   Minden beállítás környezeti változóból jön (12-factor elv), nincs beégetett titok.
   Pl. `BANK_DROP_DURATION`, `MLFLOW_TRACKING_URI`. Miért jó: ugyanaz a kód fut lokálisan
   és Dockerben, csak a környezet más. A `get_settings()` gyorsítótáraz, hogy ne olvassuk
   újra feleslegesen.

2. **`logging_setup.py` — strukturált (JSON) naplózás.**
   Minden naplósor egy JSON objektum (idő, szint, üzenet). Miért: géppel is olvasható, és
   később a predikciós kérések naplója lesz a drift-riport bemenete. A beállítás
   *idempotens*: akárhányszor hívjuk, nem keletkezik duplikált naplókezelő.

3. **`schemas.py` — adat-sémák (pydantic).**
   A `ClientRecord` írja le, milyen bemenetet fogad el a `/predict`. Fontos döntés: a
   `duration` mező **nincs** benne — mert az csak a hívás *után* ismert, tehát "beszivárogna"
   a célváltozó (adatszivárgás). Az `extra="forbid"` miatt ismeretlen mezőt is elutasít
   (biztonságos határvédelem).

**Tesztelés:** mindhárom modulra 100% lefedettség, az összes ág (pl. hibás érték → elutasítás,
kivétel-ág a naplózóban) külön tesztelve. A tesztkód is beleszámít a coverage-be.

**Tanulság:** a jó rendszer a *határon* validál (rossz adat ott bukjon el), és a
beállítás/napló külön van a logikától — így tesztelhető és cserélhető.

---

## Fázis 2 — Adat, előfeldolgozás, modell

**Cél:** a nyers CSV-ből tanítható modell, de a lépések külön, tesztelhető darabokban.

Négy modul, mind egy felelősséggel:

1. **`data.py` — betöltés és előkészítés.**
   - `load_data`: a `;`-elválasztott CSV beolvasása.
   - `feature_columns`: megadja a numerikus és kategorikus oszlopokat. A `duration`
     alapból **kimarad** (adatszivárgás), de kapcsolóval visszakérhető.
   - `prepare_xy`: szétválasztja a jellemzőket (`X`) és a célt (`y`), a `yes/no`-t
     `1/0`-ra fordítja, és hibát dob ismeretlen célértéknél.
   - `split`: **rétegzett** (stratified) tanító/teszt vágás — megőrzi az osztályarányt
     (fontos, mert a "nem" ~88%).

2. **`features.py` — előfeldolgozó (`ColumnTransformer`).**
   Numerikus → standardizálás; kategorikus → one-hot kódolás `handle_unknown="ignore"`-ral,
   hogy egy sosem látott kategória se dobjon hibát kiszolgáláskor.

3. **`model.py` — a modell egyetlen `Pipeline`-ban.**
   Az előfeldolgozó + osztályozó (`HistGradientBoostingClassifier`) egy csomagban. Így a
   betanított előfeldolgozás "együtt utazik" a modellel → tanítás és kiszolgálás ugyanúgy
   transzformál (nincs training-serving skew).

4. **`metrics.py` — kiértékelés.**
   Visszaad öt mérőszámot (roc_auc, f1, accuracy, precision, recall). A promóciós kapu
   (3. fázis) ezekből dönt majd.

**Egy fontos gyakorlati lecke — típusok:** a pandas/sklearn nincs teljesen típusozva, ezért
a pyright "ismeretlen" típusokat lát. A határokon **`cast()`-tel** tesszük explicitté a
típust (nem kikapcsoljuk az ellenőrzést). A `Protocol` csak típusleíró, nem fut — ezt
`# pragma: no cover` jelzi (indoklással), hogy a lefedettség tiszta maradjon.

**Állapot:** 44 teszt, 100% lefedettség, flake8 tiszta, pyright 0 hiba.

---

## Fázis 3 — Tanítás, MLflow, promóciós kapu

**Cél:** a modell tanítása *nyomon követhetően* (MLflow), és csak akkor kerüljön élesbe,
ha nem rontja a korábbi eredményt.

Három modul:

1. **`gate.py` — promóciós kapu.**
   Egy tiszta döntési szabály: az új ("candidate") modell csak akkor válthatja le a
   jelenlegi éleset ("Production"), ha a választott mérőszámban (alapból `roc_auc`) nem
   rosszabb (egy tűréshatáron belül). Ha nincs még éles modell, az új mehet. Ez a feladat
   kulcskövetelménye: **automatikusan megakadályozza a romlást**.

2. **`registry.py` — MLflow burkolók.**
   - `setup_tracking`: beállítja a tracking szervert és a kísérletet.
   - `log_training_run`: elindít egy "run"-t, logolja a paramétereket, mérőszámokat, és
     elmenti + regisztrálja a modellt (artifact). Visszaadja a run azonosítót és a verziót.
   - `load_production_pipeline`: betölti az éles modellt (vagy `None`-t, ha nincs).
   - `promote_model`: a kiválasztott verziót `Production`-ra állítja, a régit archiválja.
   A model registry **adatbázis-backendet** igényel (SQLite/Postgres), sima fájltár nem elég.

3. **`train.py` — összefűzés.**
   Betölt → vág → tanít → kiértékel → lekéri az éles modell eredményét ugyanazon a teszten →
   a kapu dönt → MLflow-ba logol és regisztrál → ha a kapu enged, promótál és frissíti az
   Evidently referencia-adatot. `main()` a `bank-train` konzolparancs.

**Gyakorlati leckék (amibe beleszaladtunk):**
- **Verzió-ütközések:** az `mlflow` lehúzta a `pyarrow`-t 17-re, és a túl friss
  `SQLAlchemy` (2.1) hiányzó osztály miatt elszállt — ezért **pontos verziókra pinneltünk**
  (`sqlalchemy==2.0.35`). Ez pont az, amiért kellenek a reprodukálható `requirements`.
- **Coverage + alfolyamat:** az MLflow a modell mentésekor egy segéd-alfolyamatot indított,
  ami összezavarta a lefedettségmérést. Megoldás: explicit `pip_requirements` megadása, így
  nem indul az alfolyamat.
- **Tesztelés szerver nélkül:** a registry teszt egy **helyi SQLite** MLflow-tárat használ
  ideiglenes mappában — valódi integráció, de nem kell hozzá szerver. A `train` tesztjei az
  MLflow-hívásokat **mockolják** (így gyorsak és determinisztikusak).

**Állapot:** 56 teszt, 100% lefedettség, flake8 tiszta, pyright 0 hiba.

---

## Fázis 4 — Predikció és kérés-naplózás

**Cél:** az éles modellel predikciót adni, és minden kérést + eredményt naplózni (ez lesz
a drift-figyelés bemenete).

- **`logstore.py`** — `RequestLogStore` protokoll + `JsonlFileLogStore`: minden predikció
  egy JSON sor. Később ezt olvassa vissza a monitoring. (A tesztekben egy memóriabeli
  "Fake" store-t használunk — gyors és ellenőrizhető.)
- **`predict.py`** — `Predictor`: betölti az éles pipeline-t, a bejövő rekordot a
  `ClientRecord` sémával validálja, predikál (címke + valószínűség + modellverzió), és
  naplóz. `load_predictor` összerakja az éles modellből; ha nincs éles modell,
  `ModelNotReadyError`-t dob (előbb tanítani kell).

**Tanulság:** a predikciós út is a *határon* validál, és minden kérést eltesz — enélkül
nem lenne mit monitorozni. A betölött modell "egyben" jön (pipeline), így a kiszolgálás
ugyanazt az előfeldolgozást használja, mint a tanítás.

**Állapot:** 61 teszt, 100% lefedettség, flake8 tiszta, pyright 0 hiba.

---

## Fázis 5 — REST API (FastAPI)

**Cél:** a modell elérhető legyen HTTP-n keresztül, `train` és `predict` végponttal.

- **`api/app.py`** — alkalmazás-gyár (`create_app`) + hibakezelő: ha nincs éles modell,
  `503`-at ad (nem 500-at). A `uvicorn bank_marketing.api.app:app` így indítható.
- **`api/routes.py`** — három végpont:
  - `GET /health` → `{"status":"ok"}` (életjel, konténer healthcheckhez).
  - `POST /predict` → a bemenetet a `ClientRecord` validálja (rossz adat → `422`),
    majd predikció + naplózás.
  - `POST /train` → elindít egy tanítás+kapu ciklust; opcionális `test_size`/`random_state`.
- **`api/deps.py`** — függőség-injektálás: a predikciószolgáltató külön kis függvény, amit a
  tesztek felül tudnak írni (így nem kell MLflow a teszthez).

**Tanulság:** a végpontok "vékonyak" — csak hívják az `src` logikát. A tesztek FastAPI
`TestClient`-tel mennek, a modellt/MLflow-t Fake/Mock váltja ki → gyorsak és izoláltak.
A `503` külön státusz jól jelzi, hogy "még nincs betanított modell", nem pedig belső hiba.

**Állapot:** 68 teszt, 100% lefedettség, flake8 tiszta, pyright 0 hiba. A teljes Python
alkalmazás kész és tesztelt; jöhet a konténerizálás.

---

## Fázis 6 — Docker Compose (egyetlen paranccsal indul)

**Cél:** `docker compose up` indítsa az egész rendszert.

Szolgáltatások:
- **postgres** — az MLflow backend-adatbázisa (a model registry adatbázist igényel).
- **mlflow** — tracking szerver + model registry; a UI a **5001** host-porton (a 5000-et a
  macOS AirPlay foglalja).
- **api** — a FastAPI alkalmazás (8000). A `./data` kötetként be van fűzve, így a naplók és a
  referencia-adat a hoston is látszanak.

A konténerek a compose-hálózaton **szolgáltatásnév** alapján érik el egymást
(`mlflow:5000`, `postgres:5432`). A `depends_on` + healthcheck garantálja a helyes sorrendet.

**Gyakorlati leckék (amibe beleszaladtunk):**
- **Céges registry-proxy:** a Docker Desktop csak a hivatalos "library" image-eket engedi
  (postgres igen, MinIO nem; a quay.io tiltott). Ezért **elhagytuk a MinIO-t**: az MLflow az
  artifactokat egy **megosztott köteten** tárolja és a szerveren keresztül szolgálja ki
  (`--serve-artifacts`). Ugyanaz a funkció, S3 nélkül — a feladat engedi az indokolt eltérést.
- **Foglalt port:** a 5000-et a macOS AirPlay használja → a host-oldali portot 5001-re tettük.

**Eredmény:** `docker compose up` után a `/train` végigfut (MLflow-ba logol, modellt regisztrál
és promótál), a `/predict` predikál és naplóz — minden konténerben, egyetlen paranccsal.

**Állapot:** 68 teszt, 100% lefedettség; a rendszer egy paranccsal indul és működik.