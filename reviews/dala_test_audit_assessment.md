# DaLA – foreløbig kvalitetsscreening af test-audit (48 par)

**Formål:** Kvalitetskontrol af benchmarksætninger, ikke valg/justering af prompt eller model.
Denne screening er foretaget af en enkelt AI-assistent. Den er ikke blindet, adjudikeret eller verificeret af en uafhængig dansk sprogekspert.

## Opgørelse

- **48** par i test; præcis **3 pr. kategori** fordelt på **16 kategorier**.
- Indført fejl plausibel: **47/48**.
- Reference uden specifikt identificeret forbehold: **38/48**.
- Referencer med forbehold: **9/48**.
- Reference vurderet muligvis direkte forkert: **1/48**.
- Mulig alternativ gyldig rettelse: **7**, usikker: **2**.
- Opfølgningsmarkeringer (union af tvivl og alternativer): **11**.

## Rækker til prioriteret opfølgning

| Pair ID | Problemtype | Vurdering |
|---|---|---|
| `test-00233` | `reference_missing_final_punctuation` | Den målrettede ændring fler→flere er korrekt. Referencen er en fuld sætning uden slutpunktum; muligvis kildeafhængig tekstudtrækning. |
| `test-00256` | `reference_introduces_genitive_error` | Hvis egennavnet er 'Odd-Catla', er 'Odd-Catlas' korrekt dansk genitiv. Apostroffen i 'Odd-Catla's' er normalt forkert; mulig omvendt/ugyldig korruption. Verificér navnet i originalkilden. |
| `test-00683` | `possible_residual_reference_grammar` | Målrettelsen opfordrer→opfordre er korrekt. 'Politi kørte rundt' virker usædvanligt uden bestemt form ('Politiet'); afhænger af oprindelig kontekst og journalistisk sprogbrug. |
| `test-00093` | `underdetermined_deletion_and_reference_format` | 'Cifre' er plausibelt, men ikke eneste mulige ord; udtrykket er fragmentarisk uden kontekst. 'VMs' kan desuden være en uregelmæssig genitiv af forkortelsen VM (typisk 'VM's'). |
| `test-00758` | `alternative_preposition` | 'Under Anden Verdenskrig' er grammatisk og semantisk plausibelt; 'i Anden Verdenskrig' kan også være en gyldig minimal udfyldning. |
| `test-00873` | `underdetermined_deletion_awkward_reference` | Flere verber kan udfylde det manglende led. 'Bemærkes ... med stor opmærksomhed' er mindre idiomatisk end fx 'følges ... med stor opmærksomhed'. |
| `test-00171` | `missing_antecedent_gender` | 'Han' er en mulig rettelse, men ingen entydig person/antecedent findes i den isolerede sætning; også 'hun' kan passe ved en anden kontekst. |
| `test-00551` | `other_reference_punctuation_error` | Målrettelsen et→en er korrekt, men der mangler efter alt at dømme komma efter relativsætningen: 'Det der lignede en flugt, havde ...'. Andre tegnsætnings-/formuleringsændringer kan være valide. |
| `test-00552` | `sentence_fragment_context` | Målrettelsen et→en er korrekt, men sætningen begynder med 'Lirker' uden eksplicit subjekt; kan være et løsrevet tekstuddrag. Fuld grammatiskhed afhænger af tekstgenre/kontekst. |
| `test-00739` | `possible_residual_compound_spelling` | Ombytningen 'år år i'→'år i år' er rettet. 'Mozart år' forekommer som en delt sammensætning; 'Mozartår' eller 'Mozart-år' er mere almindelige stavemåder. |
| `test-00195` | `other_reference_typography_error` | Relativordet der→som er korrekt, men 'told-og' mangler tilsyneladende mellemrum efter bindestregen: 'told- og ...'. Reference kan derfor belønne en urettet skrivefejl. |

## Afgrænsning og konsekvenser

1. **Test-data må ikke bruges til prompttilpasning.** Auditen anvendes alene til at bedømme datasættets validitet og rapportere forbehold; main experiment anvender fortsat alle test-rækker, medmindre en forud dokumenteret eksklusionsregel besluttes før modelresultater ses.
2. **Den vigtigste mulige fejlkonstruktion er `test-00256`.** Dansk genitiv af et navn, der ender på -a, skrives normalt direkte med -s, ikke apostrof+s. Kontrollér egennavnet i kildekorpus før endelig afgørelse.
3. **`delete` er ofte ikke entydig.** Det er muligt at have en korrekt rettelse, der afviger fra den eneste reference. Fortolk Exact Match som referenceoverensstemmelse og gennemfør manuel vurdering af modeludvalg.
4. **Kontrolsætninger kan indeholde restfejl/tekstfragmenter.** Overcorrection-rate betyder derfor andel ændrede *oprindelige referencer*, ikke nødvendigvis andel fejlagtige ændringer af objektivt fejlfri sætninger.
5. **Reproducerbarhed:** Sammenhold alle dataset-/output-hashes med `dataset_manifest.json`; den indsendte auditfil beviser ikke i sig selv at DaLA-revisionen er fastlåst.
6. **Usikkerhed:** 3 observationer pr. kategori er en feasibility-screening, ikke et præcist estimat på fejlprocenten i de 1.024 testpar.

## Kolonner og labels

- `error_plausible=yes`: target-input har en identificerbar sandsynlig sproglig fejl; `no`: ingen faktisk fejl kunne bekræftes.
- `reference_appropriate=yes`: ingen specifik fejl identificeret; `qualified`: tilsigtet edit ofte korrekt, men hele referencen eller entydigheden kræver forbehold; `no`: muligvis direkte forkert reference.
- `alternative_valid_correction=yes/uncertain/no`: er der en anden plausibel rettelse i forhold til referencen?
- `needs_follow_up=yes`: kandidat til manuel adjudikation. Originale sætningstekster er ikke ændret.

## Eksterne retskrivningskilder (afgrænset verificering)

- Dansk Sprognævn / sproget.dk: Genitiv (ejefald) – https://sproget.dk/typiske-problemer/genitiv-ejefald/
- Retskrivningsordbogen § 19.5: “nogensinde” eller “nogen sinde” – https://ro.dsn.dk/?paragraf=19&type=rulesearch

SHA-256 af den indsendte auditfil: `70a1a5e345848f615607f10f1a7f9dbc2dbf01bd6b5d91c32f37a05525134f27`.
