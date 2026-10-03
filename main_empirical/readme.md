
| Step | What's inside | 
|---|---|
| step1 | CS/language/year/type/abstract-validity filters; information required downstream | 
| step2 |  `tf`; per-paper unique word lists for NLTK nouns, spaCy nouns, and spaCy noun chunks | 
| step3 | Three-field counts for all valid authors, unique-max labels, in-migration counts, and active author counts | 
| step4 | `tf`, vocabulary 1000, `k=5/10/15`, count aggregation and Gini | 
| step5 | Primary-topic proportions for the three fields; heatmap `data` and `y_labels` | 
| step6 | Three noun definitions; Full/AI/NonAI; state needed for the DF threshold, new words, temporary deaths, and Full survival counts | 
| step7 | The uploaded version's cited-year histogram and Gini | 
