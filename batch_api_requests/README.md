# Usage

## OpenAI Batch API Generation for Warm-Start SFT

We consider warm-starting the synthetic data generator with a small number of expert demonstrations to obtain a reasonable starting policy for RL.

We use the following steps to (1) create batch API generation requests, (2) submit those requests, (3) monitor the completion of the requests, (4) upload the resulting SFT dataset to HuggingFace.

### (1) Create Batch API Generation Requests

```
❯ uv run python /juice5/u/nband/workspace/nlingua/synth_data_gen/create_batch_api_request.py
Loaded 1000 docs from /juice5b/scr5b/nlp/data/huggingface/lingua-data/wikipedia_en_shuffled/wikipedia_en.chunk.00.jsonl from -1000 to end.
Saving 1000 prompts to sft_data/v1/gpt-5-mini-2025-08-07/docs_0000001000/active_reading/text_processing_assistant_doc_first/sft.jsonl
Saved 1000 prompts to sft_data/v1/gpt-5-mini-2025-08-07/docs_0000001000/active_reading/text_processing_assistant_doc_first/sft.jsonl
```

### (2) Submit Batch API Generation Requests

```
❯ uv run python /juice5/u/nband/workspace/nlingua/synth_data_gen/submit_batch_api_request.py
Uploading file: sft_data/v1/gpt-5-mini-2025-08-07/docs_0000001000/active_reading/text_processing_assistant_doc_first/sft.jsonl
File uploaded successfully. File ID: file-8bHXcQnBNyUYuPKMa9Awof
Submitting batch request...

Batch submitted successfully!
Batch ID: batch_68cb1d4765fc81909ba9d43a003fe2c5
Status: validating
Input File ID: file-8bHXcQnBNyUYuPKMa9Awof
Metadata: {
  "model": "gpt-5-mini-2025-08-07",
  "user": "nband",
  "run_tag": "active_reading_v1",
  "reasoning_effort": "minimal",
  "verbosity": "high",
  "user_prompt_name": "active_reading",
  "system_prompt_name": "text_processing_assistant_doc_first"
}
```

### (3) Monitor Batch API Generation Requests

If you are confident that your batch job has completed (e.g., it has been 24h), you can download the results as follows (in this case, our batch job failed so we only get an error file):

```
❯ uv run python synth_data_gen/monitor_batch_api_requests.py download-sft

Looking for completed batches...

Found completed batch: batch_68ca56923b6081908cfb2932b4a0c42c
  Model: gpt-5-mini-2025-08-07
  User: nband
  Run Tag: active_reading_v1

Found 1 completed matching batches.

Processing batch batch_68ca56923b6081908cfb2932b4a0c42c...
  Model: gpt-5-mini-2025-08-07
  User: nband
  Run Tag: active_reading_v1
Downloading error file to sft_data_out/v1/model_gpt-5-mini-2025-08-07/reasoning_effort_low/run_tag_active_reading_v1/system_prompt_name_text_processing_assistant_doc_first/user_nband/user_prompt_name_active_reading/verbosity_high/batch_68ca56923b6081908cfb2932b4a0c42c/errors.jsonl
Results saved to sft_data_out/v1/model_gpt-5-mini-2025-08-07/reasoning_effort_low/run_tag_active_reading_v1/system_prompt_name_text_processing_assistant_doc_first/user_nband/user_prompt_name_active_reading/verbosity_high/batch_68ca56923b6081908cfb2932b4a0c42c/errors.jsonl
```

Or, if you want to monitor the batch jobs with a polling interval, you can run
```
❯ uv run python synth_data_gen/monitor_batch_api_requests.py monitor

Found 2 total batches, 1 active.

Status Summary:
  in_progress: 1
  completed: 1

Active Batch Details:

==================================================
Batch ID: batch_68cb1d4765fc81909ba9d43a003fe2c5
Status: in_progress
Endpoint: /v1/chat/completions
Created at: 2025-09-17 13:42:47
Started at: 2025-09-17 13:43:49
Expires at: 2025-09-18 13:42:47

Request Counts:
  Total: 1000
  Completed: 957
  Failed: 0

Metadata: {'model': 'gpt-5-mini-2025-08-07', 'user': 'nband', 'run_tag': 'active_reading_v1', 'reasoning_effort': 'minimal', 'verbosity': 'high', 'user_prompt_name': 'active_reading', 'system_prompt_name': 'text_processing_assistant_doc_first'}
==================================================

Monitoring 1 pending batches...

==================================================
Batch ID: batch_68cb1d4765fc81909ba9d43a003fe2c5
Status: in_progress
Endpoint: /v1/chat/completions
Created at: 2025-09-17 13:42:47
Started at: 2025-09-17 13:43:49
Expires at: 2025-09-18 13:42:47

Request Counts:
  Total: 1000
  Completed: 957
  Failed: 0

Metadata: {'model': 'gpt-5-mini-2025-08-07', 'user': 'nband', 'run_tag': 'active_reading_v1', 'reasoning_effort': 'minimal', 'verbosity': 'high', 'user_prompt_name': 'active_reading', 'system_prompt_name': 'text_processing_assistant_doc_first'}
==================================================

Sleeping for 30 seconds...

[... more output ...]

==================================================
Batch ID: batch_68cb1d4765fc81909ba9d43a003fe2c5
Status: completed
Endpoint: /v1/chat/completions
Created at: 2025-09-17 13:42:47
Started at: 2025-09-17 13:43:49
Completed at: 2025-09-17 13:58:18
Expires at: 2025-09-18 13:42:47

Request Counts:
  Total: 1000
  Completed: 1000
  Failed: 0

Metadata: {'model': 'gpt-5-mini-2025-08-07', 'user': 'nband', 'run_tag': 'active_reading_v1', 'reasoning_effort': 'minimal', 'verbosity': 'high', 'user_prompt_name': 'active_reading', 'system_prompt_name': 'text_processing_assistant_doc_first'}
==================================================
Batch batch_68cb1d4765fc81909ba9d43a003fe2c5 is complete. Downloading results...
Results saved to sft_data_out/v1/batch_68cb1d4765fc81909ba9d43a003fe2c5_output.jsonl
```

### (4) Upload SFT Dataset to HuggingFace

```
❯ uv run python synth_data_gen/upload_sft_dataset_to_hf.py

Preparing dataset...

Dataset preview:
Dataset({
    features: ['input', 'output'],
    num_rows: 1000
})

Uploading to HuggingFace repository: nband/sft_active_reading_v1
Creating parquet from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.86ba/s]
Processing Files (1 / 1)                : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.57MB / 5.57MB, 4.65MB/s  
New Data Upload                         : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.57MB / 5.57MB, 4.65MB/s  
                                        : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.57MB / 5.57MB            
Uploading the dataset shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.17s/ shards]

Upload complete! You can now access your dataset on HuggingFace.
```

### (5) Optional: Get Dataset Statistics

Compute dataset statistics on an uploaded dataset, which can be helpful for figuring out SFT max sequence length, etc.

```
❯ uv run python synth_data_gen/get_dataset_statistics.py
[... more output ...]
Dataset: nband/sft_active_reading_v1  split: train  samples: 1,000

Input (chars):   count=1,000  min=495  p50=1710.50  mean=3111.83  std=4497.34  p95=10378.25  p99=19410.23  max=64,262  total=3,111,828
Input (tokens):  count=1,000  min=109  p50=440.50  mean=800.45  std=1184.60  p95=2524.55  p99=5219.44  max=19,414  total=800,447

Output (chars):  count=1,000  min=2,006  p50=6647.50  mean=7207.77  std=3138.73  p95=12842.15  p99=15721.31  max=21,721  total=7,207,774
Output (tokens): count=1,000  min=501  p50=1862.50  mean=2028.11  std=949.13  p95=3749.60  p99=4631.26  max=7,346  total=2,028,110

Combined (chars):  count=1,000  min=2,578  p50=8529.00  mean=10319.60  std=7054.69  p95=22867.25  p99=32351.65  max=83,506  total=10,319,602
Combined (tokens): count=1,000  min=621  p50=2327.00  mean=2828.56  std=1956.29  p95=6317.70  p99=9402.34  max=24,824  total=2,828,557
```

For example, above, an SFT max sequence length of 8192 would cover more than 95% of the dataset examples.