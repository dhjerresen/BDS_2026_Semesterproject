# Første manuelle screening af DaLA – 80 par

Dette dokument er en faglig, foreløbig screening af 80 rekonstruerede par fra train og val.
Markeringerne er ikke en formel benchmarkvalidering eller en blindet dobbeltannotering.

- Antal rækker: **80**; 5 i hver af 16 fejlkategorier.
- Sprogligt plausibel tilsigtet korruption: **80/80**.
- Reference uden identificeret særligt forbehold: **68/80**.
- Reference med forbehold/kræver adjudikation: **12/80**.
- Mulig alternativ gyldig rettelse: **10**; usikker: **2**.

## Prioriterede par til ekstra gennemgang

| pair_id | problemtype | bemærkning |
|---|---|---|
| `train-00173` | `other_possible_reference_error` | Målrettelsen stærker→stærkere er korrekt. Referencen indeholder 'cocain' (nutidig standard: 'kokain'); vurder særskilt norm og historisk stavning. |
| `train-00438` | `other_possible_reference_error` | Målrettelsen yderliger→yderligere er korrekt. 'maskinsationerne' ligner en urettet stavefejl (forventeligt 'maskinstationerne'). |
| `train-00257` | `other_possible_reference_error` | Målrettelsen tingende→tingene er korrekt. 'kropsduer' forekommer usædvanligt/semantisk uklart og bør verificeres mod kilden; mulig urettet fejl. |
| `train-00235` | `other_reference_punctuation_error` | Målrettelsen EF'→EF's er korrekt. Referencesætningen slutter med '85 km/t..' (dobbelt punktum). |
| `train-00236` | `underdetermined_deletion` | Den er én plausibel tilføjelse, men manglende subjekt kan også være 'det' eller et andet led afhængigt af kontekst. |
| `train-00260` | `underdetermined_deletion` | År er meget plausibelt, men det manglende substantiv kan ikke garanteres fra den isolerede sætning. |
| `train-00472` | `underdetermined_deletion` | Spørgsmål giver en korrekt sætning, men også andre substantiver kan passe; referenceordet kan ikke identificeres sikkert. |
| `val-00071` | `severely_underdetermined_deletion` | Nikke er ikke udledeligt af den isolerede sætning. Mange infinitiver er mulige; uegnet til ukritisk enkeltreference-Exact-Match. |
| `train-00130` | `sentence_fragment_or_headline` | Originalen 'Hver dansker sit klagepunkt' er en telegramagtig, verbaløs overskrift. Om den tæller som korrekt sætning afhænger af evalueringskriteriet; evt. omskrivning er mulig. |
| `val-00002` | `sentence_fragment_or_headline` | Reference er en sætning uden finit verbum ('Peter Elmegaard med ...'); plausibelt billedtekst/overskrift, ikke en fuldstændig standardsætning. |
| `val-00017` | `incomplete_discourse_context` | 'Ikke alene ...' kan normalt invitere til en efterfølgende kontrast/fortsættelse; målrettelsen hende→hun er korrekt, men konteksten er afskåret. |
| `val-00039` | `other_reference_spelling_error` | Målrettelsen der→som er korrekt i den relevante relativkonstruktion. Referencen indeholder 'rakketter', sandsynligvis stavefejl for 'raketter'. |

## Metodiske konsekvenser

1. **Ikke entydig rekonstruktion ved sletninger.** En enkelt referencesætning kan være umulig at udlede af inputtet; Exact Match kan undervurdere accepterbare rettelser.
2. **Referencer med restfejl.** Et referencepar kan have en korrekt syntetisk edit, mens andre stave- eller tegnsætningsproblemer står uændret. Det kan skævvride både samlet EM og overcorrection-målinger, hvis en model retter dem.
3. **Fuldstændighed og kontekst.** Enkelte datasætobjekter er overskrifts- eller billedtekstfragmenter, så sentence-level GEC skal defineres nøje.
4. **Scope.** Screeningen dækker **ikke** test-splittet. Før endelig eksperiment bør der laves et separat, præregistreret, stratificeret test-split-kvalitetstjek uden at anvende testeksempler til prompttilpasning.
5. **Originale kontroller.** Da kontrollerne stammer fra reference-sætningerne, bør der kontrolleres for restfejl før de betegnes som fejlfri i overcorrection-analysen.

## Fortolkning af kolonnerne

- `reference_appropriate=yes`: ingen specifikke forbehold identificeret i denne screening (ikke garanti for fejlfrihed).
- `reference_appropriate=qualified`: den målrettede edit kan være korrekt, men eksempel eller reference bør vurderes igen.
- `alternative_valid_correction=yes`: mindst én plausibel anden formulering/minimal rettelse, der kan afvige fra exact match.
- `needs_follow_up=yes`: relevant for en formel adjudikation inden undersøgelsen fastlåses.
- `issue_type`: årsag til markeret udfordring (ingen problemer kodet `none`).

## Forbehold

Alle 80 tilsigtede fejl er vurderet plausible ud fra sætningsparrene. Der er ikke gennemført uafhængig annotatorvalidering, og fuld vurdering af samtidige fejl kan være afhængig af sprognorm og oprindelig genre.
Datasetparrene er ikke ændret i annotationen; kilde- og reference-tekst er bevaret byte-for-byte ved CSV-export (Unicode-indhold).
