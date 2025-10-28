## LO2 explainability scan (2025-10-28 07:08:07Z)

- Directory: `/Users/MTETTEN/Projects/LogLead/demo/result/lo2/explainability`
- Required files present: if_nn_mapping.csv, if_false_positives.txt
- Missing artefacts: none
- SHAP artefacts: 5 file(s) (.png: 4, .txt: 1)
- False positives list: 43938 listed
- Metrics:
  * metrics_dt.json: accuracy=0.9999, aucroc=0.9999, f1=0.9999, support=386298.0000
  * metrics_lr.json: accuracy=0.7552, aucroc=0.8480, f1=0.7378, support=386298.0000
  * metrics_seq_lr.json: accuracy=1.0000, aucroc=1.0000, f1=1.0000, support=137.0000

### SHAP files
- dt_shap_bar.png
- dt_shap_summary.png
- lr_shap_bar.png
- lr_shap_summary.png
- seq_lr_shap_skipped.txt

### False positive preview
- row_id=16 | seq_id=light-oauth2-data-1719947646__correct__light-oauth2-oauth2-client-1 | service=light-oauth2-oauth2-client-1 | score_if=0.427581
- 19:00:39.382 [XNIO-1 task-2]  s_E1ijrbRQOAKK-720CZ1Q DEBUG c.networknt.schema.RequiredValidator debug - validate( {"clientType":"public","clientProfile":"mobile","clientName":"45a92770-187e-47ca-b72e-dbd29a37","clientDesc":"1bc868bc-5be4-44ef-86b1-bf1b4ee2d341","scope":"read write","redirectUri":"http://localhost:8000/authorization","ownerId":"admin","host":"lightapi.net"}, {"clientType":"public","clientProfile":"mobile","clientName":"45a92770-187e-47ca-b72e-dbd29a37","clientDesc":"1bc868bc-5be4-44ef-86b1-bf1b4ee2d341","scope":"read write","redirectUri":"http://localhost:8000/authorization","ownerId":"admin","host":"lightapi.net"}, requestBody)
- row_id=17 | seq_id=light-oauth2-data-1719947646__correct__light-oauth2-oauth2-client-1 | service=light-oauth2-oauth2-client-1 | score_if=0.436808
- 19:00:39.382 [XNIO-1 task-2]  s_E1ijrbRQOAKK-720CZ1Q DEBUG com.networknt.schema.TypeValidator debug - validate( "http://localhost:8000/authorization", {"clientType":"public","clientProfile":"mobile","clientName":"45a92770-187e-47ca-b72e-dbd29a37","clientDesc":"1bc868bc-5be4-44ef-86b1-bf1b4ee2d341","scope":"read write","redirectUri":"http://localhost:8000/authorization","ownerId":"admin","host":"lightapi.net"}, requestBody.redirectUri)
- row_id=18 | seq_id=light-oauth2-data-1719947646__correct__light-oauth2-oauth2-client-1 | service=light-oauth2-oauth2-client-1 | score_if=0.436808

### Top feature lists
- dt_top_trigrams.txt (rows=20)
  * 1. :00
  * 2. 29:
  * 3. 00.
  * 4. 25.
  * 5. JDx
  * ...
- lr_top_tokens.txt (rows=20)
  * 1. write","redirectUri":"http://localhost:8000/authorization","ownerId":"admin","host":"lightapi.net"},
  * 2. task-1]
  * 3. requestBody)
  * 4. <init>
  * 5. c.n.openapi.ApiNormalisedPath
  * ...

