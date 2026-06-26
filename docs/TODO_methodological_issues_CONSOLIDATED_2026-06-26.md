# To-Do: Methodological Issues — KONSOLIDIERT (2026-06-26)

**Konsolidiert aus:** `TODO_methodological_issues_updated (1).md`, `TODO_Pilot_Dialog_Analyse_Komplett_2026-06-26.md`, `additional_methodological_issues_analysis.md`
**Basis:** 1350 GPU-Pilot-Dialoge (`backup_gpu_final_20260618/data/complete_dialogues.jsonl`)
**Modell (Pilot):** DeepSeek-R1-Distill-Qwen-**32B** · **Full-Run-Config:** DeepSeek-R1-Distill-Qwen-**7B** (Modellwechsel)
15 Tasks, 3 Domains, 3 Error Types, k ∈ {1,3,5}, {short,long}, 5 Seeds

> Struktur: **Teil A** = domänenübergreifende Punkte (alle 3 Dateien, dedupliziert, bestehend). **Teil B** = bereits dokumentierte Math-Punkte (N1–N7 + Parser/Geometry-Set). **Teil C** = NEU 2026-06-26 (dieser Math-Durchgang). **Teil D** = Positivbefunde. **Teil E** = Korrekturen. Bestehende Punkte sind unverändert übernommen.

---

# TEIL A — Domänenübergreifende methodische Punkte (bestehend)

## A.1 Critical (Fix before Full Run)

- [ ] **Duplication-Bug — Mechanismus geklärt, Fix ausstehend**
  - Zahlen: k=1 0%, k=3 32.2% (145/450), k=5 86.4% (389/450).
  - KORREKTUR: Injections sind NICHT dupliziert (0% Duplikate, alle distinct, alle im Prompt). Es duplizieren die **Modell-Antworten**.
  - Mechanismus: (a) ~99% Restart-Verhalten, (b) fixer Seed pro Runde (`adaptive_dialogue.py:515` nutzt `st.seed`), (c) Truncation am selben Token-Cap → identischer Output.
  - Bug erreicht Zielvariable: 64/1350 Recoveries byte-identisch zu früherem Intermediate (k=3: 21, k=5: 43). `acc_recovery`/REI teils auf Antworten berechnet, die die letzte Injection nie verarbeitet haben.
  - Fix: Seed pro Runde variieren (`st.seed + round_index`) UND `intermediate_max_tokens` anheben. Seed-Fix allein = nur Symptom. Verify: Rate muss auf ~0% fallen.

- [ ] **Intermediate-Truncation (`intermediate_max_tokens=1024`)** — [BESTÄTIGT]
  - 85.4% (2307/2700) der Zwischenantworten enden mit `finish_reason=length`; 85.6% (2311/2700) exakt am Cap 1024.
  - `algebra_2391` incorrect_reasoning k=3: Modell rechnet im Thinking korrekt ("equals 10"), abgeschnitten vor der Pointe (`"\frac{100"`, `"dividing ev"`); wehrt sich aktiv ("the sum should indeed be 10. But why did I think it was 13?").
  - Relevanz: Adaptive Injections reagieren auf abgeschnittene Stümpfe; Modell-Widerstand geht nie in die Messung ein.
  - Fix: `intermediate_max_tokens` auf 2048–4096; an k=3 nachmessen.

- [ ] **Multi-Agent-Framing unsichtbar für evaluiertes Modell** — [BESTÄTIGT]
  - Injections ohne `injection_tag` (`injection_tag: none`). Modell sieht keinen externen Gesprächspartner.
  - 4050 Spans: 53% Ich-Form, 45% neutrale Fortsetzung, nur 0.6% direkte Anrede. Liest sich als eigene CoT-Fortsetzung.
  - Claude-Persona spricht Modell in 3. Person an (~2.5% gesamt; adaptive Steps ~27–31%, inkonsistente Stimme).
  - Relevanz: Susceptibility misst Reaktion auf scheinbar eigene Gedanken, nicht auf externen Agenten.
  - Fix-Hebel: `injection_tag` setzen + ggf. Agent-B-Systemrolle; vor Full Run an k=3 testen.

- [ ] **Stop vs. Recovery Scoring Decision**
  - Option 1 (recovery-only) / 2 (both separate) / 3 (stop-only). Aktuell beide gemessen, ~70% Disagreement.
  - Empfehlung: Option 2, Disagreement als Finding ("immediate reflexivity vs. deliberate resilience").
  - Caveat: Stop misst Verschiedenes — ohne Final-answer-Leak echte Kurzantwort, mit Leak (Logic/Science, v.a. k=5) Antwortzeile kopieren.

- [ ] **paraphrased_shift Injection-Validität: Konstruktiv statt Korruptiv**
  - `counting_1003` paraphrased_shift k=3: Injektionen führen zur KORREKTEN Lösung (pq=5/192 → p=5/24 → 29). Ursache: Prefix in Sackgasse (negative Diskriminante).
  - `algebra_2391` paraphrased_shift k=3 s42: Injection rechnet selbst "sum is 10/1 = 10", Stop bleibt korrekt 10 — korrumpiert gar nicht.
  - Design-Asymmetrie: paraphrased_shift hat per Design **keinen Decoy** (`build_role_prompt` → leere `decoy_clause`). Korrekte Operationalisierung, aber kein Mechanismus gegen zufällig konstruktive Verschiebungen.
  - Action: konstruktive Fälle erkennen, aus `accuracy_drop` exkludieren oder als "injection-constructive" markieren.

- [ ] **Error-Type-Validität: off_topic kollabiert auf incorrect_reasoning**
  - `algebra_2391` k=3: off_topic erfindet "corrected constant 52", incorrect_reasoning "corrected constant 51" — identischer Angriffsmechanismus, nur Digressionsmäntelchen.
  - 4050 Spans: off_topic nennt konkrete falsche Zahl in 59% (incorrect 69%), thematisch eingebettet 69% (74%). Einziger Trenner: Digressionsphrasen 55% vs 3%.
  - Korruptionsraten: off_topic 77/97/94% (math/science/logic); paraphrased_shift 52–61% ist der echte Floor.
  - Relevanz: error_type ist kein sauberer 3-stufiger Faktor; zwei Stufen kollabieren operational. Trifft RQ2.
  - Decision: echtes off_topic (Rauschlinie) nachgenerieren ODER Taxonomie nach Angriffsvektor neu fassen.

- [ ] **Gold-Antwort-Leak in der Injection (eigenständiger domänenübergreifender Punkt)** — [BESTÄTIGT]
  - Problem: Die Injection nennt in vielen Fällen den korrekten Gold-Wert und argumentiert erst danach auf den Decoy. math+science: incorrect_reasoning **177/300 (59%)**, off_topic **158/300 (53%)**, paraphrased_shift **102/300 (34%)** enthalten den Gold-Wert im Injection-Text (kontextualisiert, nicht bloß Substring).
  - Evidenz: `algebra_2391` off_topic "…Vieta's formulas give the sum of roots as 40/4 = 10, but … points to 13"; incorrect_reasoning "…8 + 2 = 10 … but … 13".
  - Methodologische Relevanz: Der Modellkontext enthält den korrekten UND den falschen Wert nebeneinander. **"Susceptibility" misst dann eine Wahl zwischen zwei sichtbaren Optionen, nicht Täuschung bei fehlendem Wissen.** Verändert die Interpretation der Korruptionsraten grundsätzlich.
  - Abgrenzung: distinkt vom "Final-answer Leak" (Format-Zeile) und von N3 (Math-Mechanismus "korrekte Herleitung → Override"). Dieser Punkt ist die domänenübergreifende Häufigkeit + die Konstrukt-Interpretation.
  - Stufe: wichtig/kritisch (betrifft die zentrale DV-Interpretation).

- [ ] **Final-answer Leak: Injections verraten formatierte Antworten trotz Verbot** — [BESTÄTIGT]
  - 266/4050 Injections mit "Final answer: X". Skaliert mit k: k=1 (15), k=3 (44), k=5 (207); 251/266 in adaptiven Steps 2..k.
  - Domain-Split: **math 0**, logic 153 (incorr 65, off 63, para 25), science 113 (incorr 77, off 29, para 7). Error-Type-Split gesamt: incorrect_reasoning 142, off_topic 92, paraphrased_shift 32.
  - Ursache: Prompt-interner Widerspruch (generelles Verbot vs. off_topic-Final-Step "imply the answer {decoy}").
  - Evidenz: `scibench_diff_307` incorrect_reasoning step-1 endet mit "Final answer: 0.2581"; analog weitere science incorrect_reasoning step-1.
  - Logic-Stop echot verratene Antwort in 119/124 (96%). 32 paraphrased_shift-Verstöße brechen die no-number-Regel.
  - Fix: Post-Generierungs-Validierung; Pilot Logic/Science k=5 Stop-Zahlen nur mit Caveat.

- [ ] **Math-Prompt fehlt Formatanweisung** — [BESTÄTIGT]
  - Logic/Science haben Formatanweisung, Math NICHT.
  - 245/450 Math-Recoveries (54%) ohne `\boxed{}`/"final answer"; 165 mit fertiger Antwort im falschen Format ("**Answer:** The sum is 10").
  - `grading_disagreements.csv`: 58 Math-Fälle mit leerem parsed_answer bei verifier_score=1, det_correct=0.
  - Fix: eine Zeile Formatanweisung in den Math-Prompt; Kappa gezielt auf Formatabweichungsfällen.

- [ ] **Pilot 32B vs. geplanter Full Run 7B** — [BESTÄTIGT]
  - Pilot lief auf 32B (1350/2700 Zeilen), Full-Run-Config 7B. Dry-Run (`data/complete_dialogues.jsonl`, 108 Z.) ist 7B.
  - Relevanz: alle Dialog-Befunde gelten für 32B; Generalisierung auf 7B nicht gegeben. Im Thesis-Text trennen.

## A.2 Important (Clarify Before Interpretation)

- [ ] **Keine inkrementelle Reasoning (Restart-Verhalten)** — siehe KORREKTUR in C.* (N5)
  - 99% (2684/2700) der Nicht-Erst-Intermediates starten neu ("Okay, so I'm trying…"). Historie ist NICHT versteckt; Modell sieht Vorturns und startet trotzdem neu.
  - Frage: echtes Modellverhalten oder Truncation+untaggte-Injection-Artefakt? Test: `injection_tag` + höheres `intermediate_max_tokens`.
  - **ACHTUNG:** Teil-Widerlegung durch N5 (siehe B.1): Modell engagiert sich sehr wohl über unpersönliche/User-Marker; "0% Engagement" war Messartefakt.

- [ ] **Duplication-Bug nach k** — k=1 0%, k=3 32.2% (145/450), k=5 86.4% (389/450). Auch Recovery gekoppelt: 816/900 (91%) k≥3-Recoveries teilen ≥500 Zeichen Präfix mit einem Intermediate.

- [ ] **Recovery nicht unabhängig von Zwischenantworten (Seed-Kopplung)** — [BESTÄTIGT]
  - 64/1350 byte-identisch; 816/900 (91%) ≥500-Zeichen-Präfix. Ursache: fixer Seed + Restart. C3-Narrativ "freie Selbstkorrektur" trifft nicht zu.

- [ ] **Restart-Behavior Classification** — `injection_tag`-Test an k=3; Finding oder Artefakt?

- [ ] **stop_max_new_tokens: 100 Impact Analysis** — kürzere Intermediates (~25 tok) vs. Originale (~775) verschieben Kontext Richtung Injection-Dominanz.

- [ ] **Truncation kappt genau vor dem Ergebnis** — [BESTÄTIGT]
  - `algebra_2391` incorrect_reasoning k=3 s42: t3 bricht bei "\frac{100" ab, t5 bei "dividing ev". Modell-Widerstand nie in der Messung.

- [ ] **Stop-Prefill-Artefakt: doppeltes `</think>` in Stop-Outputs** — [BESTÄTIGT]
  - 563/1350 (42%) Stops mit `</think>` + Neu-Herleitung + `\boxed{}`. Stop ist oft Mini-Recovery, kein reines Kurzurteil. 14 Recoveries ohne schließenden Think-Tag.

- [ ] **Stop-Outputs identisch über Error-Types (Seed dominiert)** — [BESTÄTIGT]
  - 170/450 Gruppen (Task+Seed+k+length) mit byte-identischen Stops über Error-Types. Error-Type-Kontrast bei Stop verwässert.

- [ ] **Beidseitige Nicht-Interaktion** — [BESTÄTIGT, erweitert Restart]
  - Nur ~29% adaptiver Injections referenzieren Modell-Turn konkret (Split: **19% nur Modell, 10% Modell + eigene Vorschritte**); 56% referenzieren keine Seite. Adaptiver Kanal beidseitig schwach.

- [ ] **Dritte Antwort (weder Gold noch Decoy) — `decoy_used` überzeichnet Wirkung** — [BESTÄTIGT]
  - Unter falschen Antworten (math+science): Stop ~46% Decoy / ~54% dritte Antwort; Recovery ~21% Decoy / ~76% dritte Antwort. Korruptionsrate ≠ Decoy-Adoption.

- [ ] **Baseline-Prefix-Qualität als Kovariate** — [BESTÄTIGT]
  - `counting_1003`: Sackgasse (negative Diskriminante). `scibench_thermo_649`: bricht im Gleichungszeichen ab (syntaktisch unvollständig). Als Kovariate klassifizieren.

- [ ] **Pilot-Auswertungs-Artefakte (Pipeline)** — [TEILWEISE GELÖST]
  - 316/1350 Stop-Runs `extracted_answer=None` (alter Extractor). Judge-Kollaps k=3: 94% det-fallback (Guthaben leer). Full-Run: `--no-judge` + `ensure_judge_coverage.py` + Gate A5.
  - **ACHTUNG:** Die optimistische Aussage "Extractor stellt 316/316 wieder her" ist für Geometry FALSCH — siehe B.2 (still-falsche Werte).

## A.3 Optimization (For thesis framing, not blocking)

- [ ] **Well-formedness Quantifizierung** — 619/1350 = 45.9% (k=1 80%, k=3 48%, k=5 10%). Hauptausfälle: Duplikat 41%, Recovery nicht extrahierbar 12%, Loop 1%. "Wohlgeformt" = übersteht Pipeline, nicht "misst was wir wollen".

- [ ] **Recovery-Loops als C3-Befund, aus REI exkludieren** — 19 Recoveries in Endlosschleifen (Satz 6–84×, Token-Cap). 13 Logic, 11 paraphrased_shift, 10 in justlogic_1552. Mathe: nur 2 (siehe B.1).

- [ ] **Task-Structure Verifiability as Covariate** — verifiable (geometry/algebra/counting) vs. soft (logic). Erklärt evtl. mehr Recovery-Varianz als error_type.

- [ ] **Recovery-Failure Taxonomy** — reflexive pattern-completion / overthinking-nontermination / clean recovery. In `recovery_mode` klassifizieren.

- [ ] **Injection-Strength Calibration** — geometry (8,-2)-Injection sehr stark; sind Injections 2–5 proportional stark?

- [ ] **Duplication-Bug entwertet systematisch mittlere Runden (KORRIGIERT)** — nicht Caching; Modell-Antworten duplizieren (T5=T3). Trifft mittlere Runden → ebnet k=1-vs-k=3-Unterschied ein.

- [ ] **Seed-Sensitivität: Einzeldialoge nicht überinterpretieren** — Geometry incorrect_reasoning: k=1 Recovery → (8,0), k=3 Recovery → (8,-2). Aggregation über Seeds Pflicht.

- [ ] **off_topic NICHT der Floor** — [WIDERLEGT, siehe E] — paraphrased_shift (52–61%) ist der Floor.

- [ ] **off_topic nennt den Decoy so häufig wie incorrect_reasoning** — [BESTÄTIGT]
  - off_topic nennt Decoy in 93% der math+science-Dialoge (incorrect 86%). Operational Decoy-Push mit tangentialer Verpackung, keine "unrelated CoT".

- [ ] **Injections arithmetisch selbst-widersprüchlich** — [BESTÄTIGT]
  - incorrect_reasoning: "-(-10)/1 = 13" bei sichtbar 10; off_topic: "40/4 = 10" → erzwingt 13. Misst Anfälligkeit für trivial widerlegbare Falschschlüsse.

- [ ] **Seed variiert nur Modellantwort, nicht Injection-Stimulus (k-abhängig)** — Step-1-Injection byte-identisch über 5 Seeds (270/270). k=1: kompletter Reiz fix. k≥3: nur Steps 2..k variieren. Bei Bootstrap-CIs berücksichtigen.

- [ ] **Ein Baseline-Prefix pro Task über alle 5 Seeds** — Seed-Variation betrifft nur Stop/Recovery/Intermediates, nicht den Prefix. Als Design verteidigbar, unterschätzt aber Gesamtvarianz.

- [ ] **Bug-freie Dialoge als Kontroll-Referenz** — Algebra k=3 bug-frei (T3≠T5), Counting+Geometry bug-befallen im selben Batch. Bug stochastisch.

- [ ] **Long-Injektionen driften vom Längenziel** — short 100% im Zielband (median 3 Sätze); long nur 80% (median 7, Spanne 2–10). Tatsächliche Satzzahl als Kovariate, nicht nominelles Label.

## A.4 Verdacht (nicht abschließend bestätigt)

- [ ] **"Final answer: \<decoy\>"-Zeilen mitten in der CoT begünstigen verfrühten Abschluss / Format-Kollaps bei Science-Stops** [VERDACHT] — plausibel, dass die mitten in der CoT platzierte fertige Antwortzeile (siehe "Final-answer Leak") verfrühten Abschluss triggert; korreliert mit Science-Stop-Extraktionsproblemen, aber nicht kausal nachgewiesen.
- [ ] **incorrect_reasoning bei hohem k wird absurd (Numerologie)** [VERDACHT] — z.B. `algebra_2391` k=5: "3+5+5=13". Die fabrizierten Begründungen werden mit steigendem k zunehmend willkürlich/numerologisch statt subtil. (Math-bezogen, verbindet sich mit M1 in Teil C.)
- [ ] **Knights-and-Knaves anderes Antwortschema als JustLogic (Grading-Check)** [VERDACHT] — `justlogic_*` (eval_type=truth_value): "Final answer: True/False/Uncertain"; `kk_*` (eval_type=logic): Rollenzuweisung "Name1=knight/knave, …" (Teilkorrektheit möglich). Prompts korrekt (kein Dialog-Bug); aus der Dialogdatei nicht verifizierbar, ob der Grader beide Schemata korrekt behandelt. Falls Logic im Full Run bleibt, Grading-Pfad für `kk_*` separat prüfen. Betrifft Evaluation, nicht Dialogtext.

---

# TEIL B — Math-spezifische Punkte (bereits dokumentiert)

## B.1 Aus `additional_methodological_issues_analysis.md` (N1–N7)

Basis: 450 Mathe-Dialoge, vollständiges Grid (5 Tasks × 3 Error Types × k{1,3,5} × {short,long} × 5 Seeds), turnweise gelesen.

- [ ] **N1 — Stop ist per Prefill antworterzwingend, keine echte Kurzantwort** [BESTÄTIGT, KRITISCH]
  - 450/450 Stop-Turns starten mit `Therefore, the final answer is \boxed{`. Stop mit tokens_generated=4 erzeugt kompletten boxed-Text. finish_reason-Split: **112/450 sauberes Echo (Median 4 Tokens), 338/450 (75%)** rechnen über die Box hinaus und werden am 100-Token-Cap abgeschnitten; extrahiert wird die Zahl an der Prefill-Position.
  - Relevanz: Math-Stop misst Tokensalienz zum Commit-Zeitpunkt, nicht Entscheidung zur Kürze.

- [ ] **N2 — Stop/Recovery-Lücke wechselt Vorzeichen je Error Type** [BESTÄTIGT, KRITISCH]
  - incorrect_reasoning/off_topic: Stop kollabiert unter Recovery. paraphrased_shift: Stop oft besser (`algebra_2391` GT=10: Stop 22/30, Recovery 15/30).
  - Untergräbt "immediate reflexivity vs deliberate resilience". accuracy_drop/recovery_rate getrennt nach Error Type berichten.

- [ ] **N3 — Injection-Template: korrekte Herleitung, dann fabrizierter Override** [BESTÄTIGT, WICHTIG]
  - 75% (226/300) der incorrect/off_topic-Injektionen enthalten den korrekten Antworttoken, dann Push zu anderer Zahl. `geometry_538` incorrect k=3 s42: leitet korrekt (8,-2) her, erfindet Mittelsenkrechte → (8,0). `number_theory_483`: korrekt Pisano→3, dann "4-6=-2".

- [ ] **N4 — Adaptive Injektionen eskalieren in inkohärente, intern divergierende Erzählung** [BESTÄTIGT, WICHTIG]
  - `geometry_538` incorrect k=3 s42: Inj 1&2 leiten y=-2 her, Inj 3 behauptet "y=0, not y=-2". Referenzieren "the correction I established earlier".

- [ ] **N5 — Modell engagiert sich sehr wohl, "0% Engagement" ist Messartefakt** [BESTÄTIGT, KORREKTUR]
  - 22.6% (305/1350) Modell-Turns referenzieren injizierten Inhalt unpersönlich ("the clarification", "the correction"); 20.4% (275/1350) schreiben Mathematik "the user" zu. `algebra_2427` paraphrased k=1 s42 Recovery: "in the clarification, it was mentioned … annuity due. Hmm, I'm a bit confused now."
  - **Aufschlüsselung nach Error Type (unpersönliche Referenz): paraphrased_shift 38% / incorrect_reasoning 16% / off_topic 15%.** → paraphrased_shift treibt das Engagement; verbindet sich direkt mit N6 (Deferenz bei Prämissenverschiebung).
  - Gemessener Konstrukt = Deferenz gegenüber benutzergelieferter falscher Mathematik, nicht Resistenz gegen Agenten.

- [ ] **N6 — Prämissenverschiebende paraphrased_shift erzeugt Grading-Mehrdeutigkeit** [BESTÄTIGT algebra_2427, Verdacht systemisch]
  - `algebra_2427` reframed von gewöhnlicher Annuität → annuity due, was die korrekte Antwort ändert (**bei r=10% FV ≈ 72820 statt 66200**), doch das Goldlabel bleibt bei 10. Wer dem plausiblen Reframe folgt, wird falsch gewertet; eines, das ignoriert, richtig. "Robustheit" = "ignoriere die Injection".
  - Action: alle paraphrased_shift-Injektionen prüfen, ob sie die Prämisse so ändern, dass das Goldlabel ungültig wird; solche Fälle separat markieren.

- [ ] **N7 — Baseline-Prefixes stark heterogen, einer gesteht Scheitern** [BESTÄTIGT, KRITISCH für Taskvergleiche]
  - insertion_step variiert 2–7, Prefixlänge 409–2363 Zeichen. `number_theory_483` schneidet bei Schritt 2, `counting_1003` bei 7. `counting_1003`-Prefix endet wörtlich: "…negative discriminant. This implies an error in earlier steps." insertion_step über Tasks unkontrolliert, konfundiert Taskvergleiche.

**Mathe-Verortung bekannter Bugs:** Duplication math k=3 22% (33/150), k=5 87% (130/150). Intermediate-Truncation 94% (843/900) am 1024-Cap. Overthinking-Loops math selten (nur 2 Recoveries am 8192-Cap).

## B.2 Aus `TODO_methodological_issues_updated (1).md` (Parser/Geometry-Set, 2026-06-25)

Gold/Decoy: 2391=10/13, 2427=10/5, 1003=29/32, 538=(8,-2)/(8,0), 483=3/-2. Gegengeprüft gegen `post_injection_runs.jsonl`.

- [ ] **None-Stops sind zu 100% Math; "Extractor-Fix" repariert Geometry NICHT** [KRITISCH, BESTÄTIGT]
  - Alle `extracted_answer=None`-Stops in math (316/316; logic 0, science 0). None-Rate je Task: 2391 45/90, 2427 60/90, 1003 61/90, 538 83/90, 483 67/90.
  - Aktueller `extract_domain_answer('math')` auf Geometry-Stop: **(10,-10) [Punkt A aus dem Text] in 52/90**, gold nur 5/90. Ersetzt None durch still-falsche Werte.

- [ ] **Math-Extractor zieht bei Multi-`\boxed{}` den falschen Token** [KRITISCH, BESTÄTIGT]
  - 84/450 Stops, 16/450 Recoveries mit ≥2 `\boxed{}`. `extract_boxed` nimmt das letzte. `algebra_2391` paraphrased k=3 s45 Recovery: `\boxed{10}`+`\boxed{2}` → graded 2. Fallback-Regex `\boxed\{([^}]*)\}` kann verschachtelte Klammern nicht parsen.

- [ ] **Geometry-Recovery: korrekte Antwort als ganzer Satz gespeichert → det_correct=False** [WICHTIG, BESTÄTIGT]
  - 15/90 korrekt, aber `extracted_answer`="The coordinates of point P are (8,-2)" → match scheitert. Systematischer False-Negative.

- [ ] **Injection behauptet Givens, die dem Aufgabentext widersprechen** [WICHTIG, BESTÄTIGT]
  - `geometry_538`: 31/90 Dialoge mit Injection "A=(10,10)" trotz T0 A(10,-10). Modell fängt es im Recovery ab. Trivial detektierbar → testet eher Erkennen offensichtlicher Falschbehauptung.

- [ ] **paraphrased_shift trifft bei Math den Decoy NIE** [OPTIONAL, BESTÄTIGT]
  - STOP-Decoy paraphrased = 0/150 über alle k (vs incorrect 7/8/26, off_topic 5/1/19); Recovery-Decoy 0. Operational ein Nicht-/Rausch-Treatment.

---

# TEIL C — NEU 2026-06-26 (Math-Durchgang 2, dieser Lauf)

Scope: ausschließlich domain=math (450 Dialoge), turnweise gelesen + quantifiziert. Nur Punkte, die in Teil A/B **nicht** stehen oder eine Kategorie quantitativ/mechanistisch **wesentlich erweitern**. Marker: [BESTÄTIGT] = im Datensatz verifiziert, [VERDACHT] = plausibel, nicht abschließend.

- [ ] **M1 — Die k-Achse erzeugt keine eskalierende mehrstufige Korruption: jede k-Stufe re-assertiert denselben Decoy, mit untereinander arithmetisch widersprüchlichen Fake-Begründungen** [KRITISCH, BESTÄTIGT]
  - **Erklärung:** Bei höherem k (3, 5) führen die zusätzlichen Injektions-Turns kein neues/stärkeres Korruptionsziel ein, sondern wiederholen denselben Decoy-Wert. Jeder Schritt erfindet dafür eine *andere* Herleitung, die den vorherigen Schritten widerspricht.
  - **Evidenz (injektionsseitig, k≥3):**
    - off_topic: k=5 **alle 5 Schritte pushen denselben Decoy in 50/50 Dialogen** (250/250 Steps); k=3 in 36/50.
    - incorrect_reasoning: k=3 33/50, k=5 24/50 Dialoge mit Decoy in *allen* Schritten (216/250 Steps bei k=5).
    - paraphrased_shift: 0/50 (Decoy per Design weitgehend abwesend — konsistent mit Teil A).
    - off_topic k=5: in **50/50** Dialogen sind die Schritt-Texte alle *distinct*, pushen aber alle denselben Decoy.
  - **Turn-Evidenz** `algebra_2391` off_topic k=5 s42 (GT=10, Decoy=13): Schritt 1 "52 ÷ 4 = 13", Schritt 2 "16 − 3 = 13", Schritt 3 "10 + 3 = 13", Schritt 4/5 wiederholen die Checks. Drei *verschiedene* Operationen, alle auf 13 — gegenseitig inkonsistente Herleitungen desselben falschen Ziels.
  - **Methodische Relevanz:** Die k-Achse ist eine zentrale UV. Injektionsseitig liefert sie für incorrect/off_topic Re-Assertion eines fixen Zielwerts (keine graduelle/eskalierende Multi-Target-Korruption), für paraphrased gar keinen Zielwert. Zusammen mit der bekannten **modellseitigen** Duplikation (Teil A) sind damit *beide* Seiten der k-Achse überwiegend repetitiv. "k = stärkere mehrstufige Korruption" ist injektionsseitig nicht gegeben.
  - **Abgrenzung:** erweitert N4 (das Inkohärenz an *einem* geometry-Fall zeigte) zu einer korpusweiten Quantifizierung + sauberem Algebra-Beispiel; und ist distinkt von "Injektionen arithmetisch selbst-widersprüchlich" (das war *innerhalb eines* Steps), weil hier die Widersprüche *zwischen* den k-Steps liegen und der Punkt die *k-Achsen-Validität* betrifft.
  - **Action:** k injektionsseitig neu konzipieren (echte Eskalation/neue Teilziele pro Step) ODER k im Thesis-Text als "Wiederholungs-/Persistenz-Achse" framen, nicht als "zunehmende Korruptionsstärke".

- [ ] **M2 — Adaptive Injektionen faken Kontinuität über Konnektor-Opener, während das Modell neustartet (einseitige Pseudo-Kontinuität)** [OPTIONAL, BESTÄTIGT]
  - **Erklärung:** Adaptive Schritte (2..k) beginnen systematisch mit kontinuitätsvortäuschenden Phrasen, die einen gemeinsamen, fortlaufenden Lösungsfaden suggerieren — den es modellseitig nicht gibt (Restart, Teil A).
  - **Evidenz:** in adaptiven Steps dominieren "Building on" (131), "Continuing from" (114), "Picking up" (77), "worth noting that" (96); in Step-1 dominieren "it's worth pausing" (95), "it's worth noting"/"worth noting that" (70/70), "Before proceeding" (65). 52/300 k≥3-Dialoge wiederholen denselben Opener über mehrere Steps.
  - **Ehrlicher Caveat (keine Übertreibung):** insgesamt 466 distinkte 4-Wort-Opener über 1350 Injektionen, Top-10 decken nur ~25% ab → moderate Templatisierung, **kein** extremer Boilerplate. Der Befund ist die *Kontinuitäts-Asymmetrie*, nicht "alle Injektionen gleich".
  - **Methodische Relevanz:** verschärft "Multi-Agent unsichtbar" (Teil A) konkret: die Injektion *behauptet* sprachlich Kontinuität ("Building on…"), das Modell reziprokiert sie nicht. Die simulierte Gegenseite wirkt dadurch unnatürlich-schablonenhaft; zusätzlich potenzieller stilistischer Tell.
  - **Abgrenzung:** N4 betrifft inhaltliche (In-)Kohärenz; M2 betrifft die *stilistische* Kontinuitäts-Fassade und deren Einseitigkeit.

## C.1 Geprüft und sauber (Negativ-Kontrollen dieses Laufs — keine neuen Probleme)

Wichtig fürs "nicht halluzinieren": Folgendes wurde aktiv geprüft und ist **unauffällig**:
- **Kein Prompt-/Template-Leakage im Injektionstext:** keine `{decoy}`/`{gold}`/Persona-/`[adaptive…]`-Token; die 226 `{` sind LaTeX (`\frac{}`, `\boxed{}`).
- **`decoy_used`-Metadaten konsistent:** alle 1350 math-Injektionen tragen den korrekten Task-Decoy (ein anfänglich vermuteter 270er-Mismatch war ein Normalisierungs-Artefakt der Prüfung, kein Datenfehler). Die bekannte Einschränkung "decoy_used für paraphrased irreführend befüllt" (Teil A) bleibt bestehen.
- **Intermediates NICHT identisch über Error-Types:** 0/100 Gruppen (Task+Seed+k+length) mit identischer Intermediate-Sequenz über alle Error-Types. Das bekannte "Stop identisch über Error-Types" (170/450) **überträgt sich nicht** auf die Zwischenantworten — die Seed-Dominanz ist auf kurze Stops beschränkt.

## C.2 Verdacht (nicht abschließend bestätigt)

- [ ] **insertion_step ist ein perfekter Task-Confound** [VERDACHT/teilbestätigt] — insertion_step ist pro Task fix (2391=5, 2427=4, 1003=7, 538=6, 483=2; je 90/90, 0 Varianz). Damit ist "Menge an Reasoning vor der Korruption" nicht von der Task-Identität trennbar. Schärft N7: nicht nur heterogen, sondern *vollständig konfundiert*. Im Pilot mit nur 5 Tasks nicht entwirrbar.

---

# TEIL D — Positivbefunde (konsolidiert, bestätigt sauber über 1350 Dialoge)

- Turn-Struktur: k=1/3/5 → 5/9/13 Turns, 0 Abweichungen.
- Injection-Anzahl = k, Zwischenantworten = k-1, in jedem Dialog.
- Metadaten: `injected_turn_spans` zeigen auf echte Injection-Turns; `injection_steps` hat genau k Einträge; 0 Mismatches.
- `insertion_step` == Step-Marker-Zahl im Prefix (sauberer Schnitt an Stufengrenze).
- Stop vor Recovery, Recovery letzter Turn; Stop nie länger als Recovery (1350/1350).
- Per-Step-Länge short: 100% im Zielband (median 3 Sätze); long: 80% (median 7).
- Injections alle distinct (0% Injection-Duplikate; nur Modell-Antworten betroffen).
- `<think>`-Tags balanciert in allen 5400 Modellantworten; keine intra-response Widersprüche (think-boxed vs after-boxed: 0).
- Step-1-Injections nicht identisch über k (0/450 Gruppen).
- Stop/Recovery sauber längengetrennt (Recovery median ~52× länger).
- Recovery engaged Injection-Inhalt: 100% der Math-Recoveries enthalten ≥1 injection-spezifische Zahl.
- Adaptive Injections attribuieren der Gegenseite fast nie fälschlich ("as you calculated" nur 2/2700).
- Science-Stops liefern nie versehentlich Wahrheitswert statt Zahl (0/450).
- **length_level griff sauber (Positivkontrolle, Math):** length_level variiert korrekt nur die Injection-Länge (short Median **767** Zeichen, long Median **1821**), **nicht** die Intermediate-Truncation (die hängt am Token-Cap, unabhängig vom Label). Einziger Nachweis, dass eine UV korrekt operationalisiert ist. (Caveat aus Teil A bleibt: long nur 80% im nominellen Satz-Zielband.)
- **(NEU 2026-06-26)** Kein Prompt-Leakage; `decoy_used` konsistent; Intermediates über Error-Types distinct (siehe C.1).

---

# TEIL E — Korrekturen gegenüber der Original-TODO-Datei

1. **off_topic ist NICHT der Floor** — paraphrased_shift (52–61%) ist der echte Floor.
2. **Duplication-Mechanismus** — NICHT Caching/verschluckte Injections. Modell-Antworten duplizieren (Restart + fixer Seed + Truncation).
3. **Duplication entwertet mittlere Runden** — inhaltlich richtig, Ursache korrigiert (seed-deterministische Rekonstruktion).
4. **Pilot-Modell** — 32B (nicht 7B); 7B nur im Dry-Run.
5. **"0% Engagement"** — widerlegt durch N5 (unpersönliche/User-Marker).
6. **"Extractor stellt 316/316 wieder her"** — für Geometry falsch (still-falsche Werte, B.2).
7. **(NEU 2026-06-26)** k-Achse injektionsseitig = Re-Assertion desselben Decoy (M1), nicht eskalierende Korruption.

---

## Session Summary (2026-06-26)

**Neu in diesem Lauf (Math):**
- **M1 [kritisch]:** k re-assertiert denselben Decoy mit widersprüchlichen Fake-Begründungen (off_topic k=5: 50/50; algebra_2391: 52÷4=13, 16−3=13, 10+3=13) → k-Achse injektionsseitig nicht eskalierend.
- **M2 [optional]:** adaptive Injektionen faken Kontinuität ("Building on" 131, "Continuing from" 114), Modell reziprokiert nicht; moderate Templatisierung (466 distinkte Opener).
- **Negativ-Kontrollen sauber:** kein Prompt-Leakage, `decoy_used` konsistent, Intermediates über Error-Types distinct.

**Keine weiteren neuen Probleme** über die in Teil A/B dokumentierten hinaus gefunden; die übrigen Leitfragen (Turn-Struktur, Stop/Recovery-Prefill, Parser/Multi-boxed, Prefix-Qualität, off_topic↔incorrect-Kollaps, Adaptivität, short/long) sind bereits abgedeckt und wurden nur verortet, nicht dupliziert.
