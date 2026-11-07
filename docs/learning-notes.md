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