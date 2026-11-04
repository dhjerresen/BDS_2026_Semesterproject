# DaLA – forhåndsfastlagt evalueringsprotokol (v1.0.0)

**Status:** Dataidentiteten og analysereglerne fastlægges med `lock_evaluation_protocol.py`, før modellen testes på test-splittet. De identificerede kvalitetsforbehold er **ikke** afgjort ved uafhængig sproglig ekspertvurdering. Modelvalg, den endelige prompt og AI Lab-opsætningen skal dokumenteres efter pilotforsøg **før** hovedeksperimentet.

## 1. Datagrundlag og adskillelse

- Kilde: `danish-foundation-models/dala`; fastlåst Hugging Face-commit `0cc96091dc265d14fb5d507114690fd85857b6da`.
- Korrektionsinput: `data/final/train.csv` (**512**), `validation.csv` (**128**), `test_errors.csv` (**1.024**). Korrekte kontrolinput: `test_controls.csv` (**1.024**).
- Testmaterialet omfatter **16** registrerede fejlkategorier, selv om DaLA også omtales som 14 nye korruptionstyper plus to fra tidligere benchmarks.
- Train bruges til promptudkast og udvikling; validation bruges til pilot og fastlæggelse af konfiguration. **Test bruges kun til den endelige evaluering**. De 48 testpar i benchmark-auditen må ikke anvendes til prompt, eksempler, modelvalg eller tuning.
- Sætningspar og kontrolinput ændres ikke, og ingen observation fjernes automatisk. Kontrolinputtene er oprindelige korpussætninger, som ikke nødvendigvis er fuldstændigt sprogligt korrekte.

## 2. Datakvalitet og sporbarhed

- Train/val: 80 screenede par; 17 markeret til opfølgning.
- Test-audit: 48 screenede par (3 pr. fejlkategori); 11 markeret til opfølgning. Registreret: 47/48 tilsigtede fejl plausible; referencer 38 `yes`, 9 `qualified`, 1 `no`. Par `test-00256` er et muligt forkert genitivreferencepar.
- Vurderingerne er udført af én AI-assistent og er **ikke** uafhængigt adjudikerede. `screening_completed=true` er kun dokumentation for gennemført screening; det validerer ikke hele benchmarket.
- `finalize_dataset.py` validerer de 48 annoterede par imod den præcise audit-stikprøve og gemmer audit-hash samt en uændret snapshotfil i `data/final/`.
- `lock_evaluation_protocol.py` verifierer filhashes, splits, par-ID'er og kontroller og skriver `evaluation_protocol.lock.json` samt foruddefinerede analyse-ID-filer. **`dataset_manifest.json` forbliver en kandidat**, indtil alle ekspertvurderinger er afklaret; protokollås er ikke det samme som uafhængig benchmarkvalidering.

## 3. Primær automatisk kvalitetsanalyse

**Population:** Alle 1.024 sætninger i `test_errors.csv` pr. model, også de 11 markerede testpar.

**Primær metrik:** *Exact Match Accuracy*, andel af outputs der matcher den tilhørende `reference_text` efter `str.strip()` på både modeloutput og reference. Det fjerner kun omgivende whitespace (og eventuelle omgivende linjeskift), **ikke** forskelle i store/små bogstaver, indre mellemrum, tegnsætning, staveform, citationstegn eller omskrivning. Ret aldrig outputs automatisk før scoringen.

- Tomme, manglende eller teknisk mislykkede svar tælles som **ikke-match** i den primære score (over alle udsendte opgaver). Fejlraten og Exact Match betinget af et vellykket svar rapporteres særskilt. Modeller med mange API-fejl skal ikke fremstå bedre på grund af fjernede observationer.
- Rapportér både **mikro-EM** (`antal matches / 1.024`) og **makro-EM** (uvægtet gennemsnit af EM i hver af de 16 fejlkategorier). Ingen af disse er direkte et mål for generel dansk sproglig korrekthed.
- For hver kategori: rapportér antal testpar, antal matches og EM. Kategorier med få observationer fortolkes varsomt. For parvise sammenligninger mellem modeller: bootstrap-forskelle på de samme testitems med en på forhånd fastlagt seed og 95 % interval, når implementeret og kontrolleret. Beskriv teknikken præcist i metodeafsnittet.

## 4. Kontrolsætninger og overcorrection

- Anvend **alle 1.024** `test_controls.csv` pr. model.
- Primært kontrolmål: *unchanged rate* = andel hvor `str.strip(output) == str.strip(input_text)`; *change rate* = 1 minus unchanged rate. Rapportér også tekniske fejl separat og specificér nævneren.
- Ændring af en oprindelig sætning er et operationelt mål for **edit propensity**, ikke automatisk en sproglig fejl: nogle originaler indeholder muligvis selv fejl. Manuel kontrol vurderer, om ændringerne faktisk var nødvendige.

## 5. Sensitivitetsanalyse – foruddefineret

- **Primær analyse ændres ikke.** Som konservativ robusthedsanalyse gentages den samlede EM-beregning på de **1.013** testpar, der **ikke** hører til de 11 på forhånd audit-markerede ID'er.
- `data/analysis_sets/sensitivity_unflagged_test_ids.csv` oprettes deterministisk af låsescriptet. Disse 11 er **flagget til opfølgning**, ikke bekræftet ugyldige; det reducerede datasæt må **ikke** kaldes et rent eller valideret benchmark.
- Rapportér om modelrangering eller effekter ændres, og hvor meget. Udeluk aldrig yderligere observationer som reaktion på modeloutput uden separat, tydeligt deklareret post hoc-analyse.
- Hvis en uafhængig sproglig adjudikation efterfølgende afklarer de 11 par, dokumentér en særskilt opdateret version af protokollen og en separat analyse; omskriv ikke denne forudregistrering.

## 6. Manuel vurdering af modeloutput

- Fast udvalg **inden der ses på modelresultaterne**: 5 tilfældigt udvalgte fejlpar pr. kategori (**80** pr. model) med seed `20261105`. For 2 af disse pr. kategori udvælges også deres korrekte kontrolvariant (**32** pr. model). Samme sætninger anvendes for samtlige modeller.
- `data/analysis_sets/manual_review_pair_ids.csv` gemmer kun ID'er og udvalgssignaler; må ikke bruges til at udforme prompten.
- Vis input, reference og modeloutput i en blindet rækkefølge, hvor bedømmeren ikke ser modelidentiteten. Vurdér **successful / partially successful / unsuccessful** på baggrund af: (a) målfejlen er rettet, (b) ingen nye væsentlige sproglige fejl, (c) betydning bevaret. Registrér separat **unnecessary_modification** (`yes/no/uncertain`) og korte begrundelser. Et output kan være gyldigt uden at være identisk med referencen.
- Grænsetilfælde eller tvetydige referencepar markeres `uncertain` og drøftes med en uafhængig fagperson, hvis det er muligt. Rapportér om kun én person har bedømt outputs, og fremlæg vurderingsusikkerhed. Manuel evaluering af modeloutput er **adskilt** fra den tidligere benchmark-kvalitetsaudit.

## 7. Latency og tekniske fejl

- Mål *end-to-end latency* med en monoton timer fra lige før request sendes, til det fulde svar er modtaget. Dette er observeret AAU AI Lab-svartid inkl. eventuel kø/infrastruktur, **ikke** isoleret modellens inferenstid.
- Foreløbig hovedregel: ét målt request pr. input og model, **ingen skjult automatisk retry i hovedkørslen**. Fejlede forespørgsler logges med status og fejltype; de tildeles ingen succesfuld responstid. Hvis transportlaget har indbyggede retries, skal de konfigureres eller dokumenteres før hovedtesten.
- Rapportér median og 90-percentil for **succesfulde** requests samt fejlrate, antal, input-/outputlængde og gennemførelsesbetingelser. Korrektioner kan have forskellig længde og påvirke svartiden. Hvis flere gentagelser gennemføres, logges hver gentagelse og analyseres særskilt; de må ikke blandes ubemærket med enkeltkørslen.

## 8. Forberedelse af AI Lab-hovedkørslen (endnu ikke fastlagt)

Før første testkald skal vi låse en *run config* med modelnavn/-version, platform/endpoint, fuldstændig prompt, temperature, max tokens, evt. seed, streaming-indstillinger, timeout, batch/concurrency, rækkefølge, latency-timer og eventuel request-retry-politik. Alle modeller får samme instruktion og alle tilgængelige indstillinger holdes ens, hvor understøttet. Hvis de tekniske indstillinger ikke kan være identiske, dokumenteres forskellene.

Logging pr. kald bør mindst indeholde: `run_id`, `model_id`, `item_id`, `pair_id`, `input_text`, `reference_text`, `output_text`, `status`, `error_message`, `latency_seconds`, `prompt_version`, genereringsparametre og tidspunkt. Sørg for ikke at indlejre hemmelige adgangsoplysninger i logs.

## 9. Begrænsninger

DaLA er syntetiske danske korruptioner, ikke tekster skrevet af personer med dysleksi. Enkelte sletninger er underbestemte, originale referencer kan have restfejl, og flere gyldige rettelser kan forekomme. EM måler overensstemmelse med **én** reference, ikke realiseret kundeværdi. Latency måles i det konkrete AI Lab-miljø. De tekniske resultater kan diskuteres i forhold til Jacoos value proposition og strategiske prioriteringer, men beviser ikke brugeroplevelse eller konkurrencemæssig fordel.
