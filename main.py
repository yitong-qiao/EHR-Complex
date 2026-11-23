import argparse
import os
import json
import logging
from pathlib import Path
import sys
import time
import toml
from ehr_gym.env.base import EHREnv
from ehr_gym.agent.base import EHRAgent
from ehr_gym.utils.general import load_config, save_conversation_history
import ray
import joblib
import contextlib
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run EHR-Gym Experiments')
    parser.add_argument('--config_path', type=str)

    parser.add_argument('--task', type=str, choices=['ehr_complex'])
    parser.add_argument('--credentials_path', type=str)
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--result_dir_tag', type=str)
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--async_run", action="store_true", help="Run experiments asynchronously")
    parser.add_argument("--n_jobs", type=int, help="Number of parallel jobs")
    parser.add_argument("--parallel_backend", type=str, help="Parallel backend to use")
    parser.add_argument("--mode", type=str, default="test", help="train/test")
    parser.add_argument("--model_name", type=str, help="Override LLM model name")
    parser.add_argument("--temperature", type=float, help="Override LLM temperature")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max_new_tokens (output token limit) in base.yaml")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p sampling; omitted by default")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling for endpoints that support it")
    parser.add_argument("--enable_thinking",  action="store_true",  default=None,
                        help="Enable thinking for supported models")
    parser.add_argument("--disable_thinking", action="store_true",  default=None,
                        help="Disable thinking for supported models")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort for GPT-5 and o-series models")
    parser.add_argument("--resume", action="store_true",
                        help="Resume evaluation, reusing scores for completed tasks")
    parser.add_argument("--rescore", action="store_true",
                        help="Rescore saved final answers and update result files in place")
    parser.add_argument("--num_param_expansions", type=int, default=None,
                        help="Maximum parameter groups per template; all by default")
    parser.add_argument("--num_paraphrases", type=int, default=None,
                        help="Maximum question variants per parameter group; all by default")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override Data.data_path with a question directory")
    return parser.parse_args()

def convert_config_to_args(config, args):
    if not args.task:
        args.task = config['task']
    if not args.credentials_path:
        args.credentials_path = config['credentials_path']
    if not args.work_dir:
        args.work_dir = config['work_dir']
    if not args.result_dir_tag:
        args.result_dir_tag = config['result_dir_tag']
    if not args.start_idx:
        args.start_idx = config['start_idx']
    if not args.end_idx:
        args.end_idx = config['end_idx']
    if not args.num_steps:
        args.num_steps = config['num_steps']
    return args

def update_dynamic_config(config, args):
    """Apply command-line overrides and derive output directory names."""
    if args.task:
        config['task'] = args.task
        config['Data']['data_path'] = f"data/{args.task}"

    if args.data_path:
        config['Data']['data_path'] = args.data_path

    if args.model_name:
        config['Agent']['llm']['model_name'] = args.model_name
        config['Agent']['llm']['model_type'] = "OpenAI"

    current_temp = args.temperature if args.temperature is not None else config['Agent']['llm']['temperature']

    if args.temperature is not None:
        config['Agent']['llm']['temperature'] = args.temperature

    if getattr(args, 'max_new_tokens', None) is not None:
        config['Agent']['llm']['max_new_tokens'] = args.max_new_tokens

    if getattr(args, 'top_p', None) is not None:
        config['Agent']['llm']['top_p'] = args.top_p

    if getattr(args, 'top_k', None) is not None:
        config['Agent']['llm']['top_k'] = args.top_k

    if getattr(args, 'enable_thinking', None):
        config['Agent']['llm']['enable_thinking'] = True
    elif getattr(args, 'disable_thinking', None):
        config['Agent']['llm']['enable_thinking'] = False

    if getattr(args, 'reasoning_effort', None) is not None:
        config['Agent']['llm']['reasoning_effort'] = args.reasoning_effort

    if args.credentials_path:
        config['credentials_path'] = args.credentials_path

    
    current_model = config['Agent']['llm']['model_name']
    safe_model_name = current_model.replace(".", "_").replace(":", "-")
    current_task = config['task']

    if not args.result_dir_tag:
        config['result_dir_tag'] = f"{safe_model_name}-{current_task}-temp{current_temp}"
    else:
        config['result_dir_tag'] = args.result_dir_tag

    if not args.work_dir:
        config['work_dir'] = f"./workdir/{safe_model_name}"
    else:
        config['work_dir'] = args.work_dir

    return config
    
def load_credentials(credentials_path):
    return toml.load(credentials_path) if credentials_path else {}

def set_environment_variables(credentials):
    for key, value in credentials.items():
        os.environ[key] = value

def create_env_config_dir(work_dir, task, result_dir_tag):
    env_config_tmp_dir = os.path.join(work_dir, task, result_dir_tag)
    os.makedirs(env_config_tmp_dir, exist_ok=True)
    return env_config_tmp_dir

def get_task_class(task):
    if task != 'ehr_complex':
        raise ValueError(f'Invalid task: {task}')
    from ehr_gym.env.task.ehr_complex import EHRComplexTask
    return EHRComplexTask


def scan_workdir_status(save_dir: str, start_idx: int, end_idx: int, task: str = None) -> dict:
    """Summarize completed, incomplete, and unreadable result files."""
    if task == 'ehr_complex':
        n_completed = n_failed = n_broken = n_total = 0
        for root, dirs, files in os.walk(save_dir):
            for fname in files:
                if not (fname.startswith('history_') and fname.endswith('.json')):
                    continue
                path = os.path.join(root, fname)
                n_total += 1
                try:
                    with open(path) as f:
                        history = json.load(f)
                    last = history[-1] if history else {}
                    if isinstance(last, dict) and 'result' in last:
                        n_completed += 1
                    else:
                        n_failed += 1
                except Exception:
                    n_broken += 1
        return {'completed': n_completed, 'failed': n_failed,
                'broken': n_broken, 'total_files': n_total}

    status = {'completed': set(), 'failed': set(), 'missing': set(), 'broken': set()}
    for idx in range(start_idx, end_idx):
        path = os.path.join(save_dir, f'history_{idx}.json')
        if not os.path.exists(path):
            status['missing'].add(idx)
            continue
        try:
            with open(path) as f:
                history = json.load(f)
            last = history[-1] if history else {}
            if isinstance(last, dict) and 'score' in last:
                status['completed'].add(idx)
            else:
                status['failed'].add(idx)
        except Exception:
            status['broken'].add(idx)
    return status


def rescore_single_experiment(args, config, idx) -> float | None:
    """Rescore a saved final answer and update its result file; return None if missing."""
    import json as _json

    save_dir_base = os.path.join(args.work_dir, args.task, args.result_dir_tag, args.mode)

    task_cls = get_task_class(args.task)
    task_kwargs = {
        'data_path': config['Data']['data_path'],
        'mode': args.mode,
    }
    if args.task == 'ehr_complex':
        task_kwargs['num_param_expansions'] = getattr(args, 'num_param_expansions', None)
        task_kwargs['num_paraphrases'] = getattr(args, 'num_paraphrases', None)

    task = task_cls(task_id=idx, **task_kwargs)
    task.setup()

    if hasattr(task, 'metadata'):
        meta = task.metadata
        save_dir = os.path.join(save_dir_base, meta['subset'], meta['category'], meta['template'])
        output_path = os.path.join(save_dir, f"history_{idx}.json")
    else:
        save_dir = save_dir_base
        output_path = os.path.join(save_dir, f'history_{idx}.json')

    if not os.path.exists(output_path):
        task.teardown()
        return None

    with open(output_path) as f:
        history = _json.load(f)

    final_answer = None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        role = item.get('role', '')
        if role == 'assistant':
            content = item.get('content', '')
            if isinstance(content, list):
                content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
            if '<Final_answer>' in content and '</Final_answer>' in content:
                start = content.find('<Final_answer>') + len('<Final_answer>')
                end = content.find('</Final_answer>')
                final_answer = content[start:end].strip()
                break

    if final_answer is None:
        logger.warning(f"[Rescore] idx={idx}: No <Final_answer> found; score is 0")
        reward = 0
    else:
        obs = {"type": "submit", "env_message": final_answer}
        reward, done, _, _ = task.validate([], obs)
    task.teardown()

    new_result = dict(history[-1]) if history and isinstance(history[-1], dict) and 'result' in history[-1] else {}
    new_result.update(result='success' if reward else 'failure', score=float(reward))
    if history and isinstance(history[-1], dict) and 'result' in history[-1]:
        history[-1] = new_result
    else:
        history.append(new_result)

    save_conversation_history(history, output_path)
    return float(reward)


def rescore_all_experiments(args, config):
    """Rescore saved answers in the selected task range and summarize scores."""
    scores = []
    skipped = 0
    for idx in tqdm(range(args.start_idx, args.end_idx), desc="Rescoring"):
        score = rescore_single_experiment(args, config, idx)
        if score is None:
            skipped += 1
        else:
            scores.append(score)

    if scores:
        accuracy = sum(scores) / len(scores)
        print(f"[Rescore] Scored {len(scores)} tasks; {skipped} result files missing")
        print(f"[Rescore] Accuracy: {accuracy:.4f}  ({int(sum(scores))}/{len(scores)})")
        log_file = os.path.join(args.work_dir, "running_records.jsonl")
        model_name = config['Agent']['llm']['model_name']
        temperature = config['Agent']['llm']['temperature']
        with open(log_file, "a+") as f:
            record_line = (
                f"[RESCORE] Task: {args.task} | "
                f"Model: {model_name} | "
                f"Temp: {temperature} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Scored: {len(scores)}\n"
            )
            f.write(record_line)
    else:
        print(f"[Rescore] No completed results found; {skipped} files missing")



def sequential_run_experiments(args, config):
    success_rate = 0
    for idx in range(args.start_idx, args.end_idx):
        success = run_single_experiment(args, config, idx)
        success_rate += success
    success_rate /= (args.end_idx - args.start_idx)

    print(f'Success Rate: {success_rate}')

def run_single_experiment(args, config, idx):
    agent_config = config['Agent']
    data_config = config['Data']
    experiment_start_time = time.time()

    save_dir_base = os.path.join(args.work_dir, args.task, args.result_dir_tag, args.mode)
    task_cls = get_task_class(args.task)
    task_kwargs = {
        'data_path': data_config['data_path'],
        'mode': args.mode,
    }
    if args.task == 'ehr_complex':
        task_kwargs['num_param_expansions'] = getattr(args, 'num_param_expansions', None)
        task_kwargs['num_paraphrases'] = getattr(args, 'num_paraphrases', None)
    env = EHREnv(task_entrypoint=task_cls, task_kwargs=task_kwargs)
    obs, info = env.reset(idx)
    if hasattr(env.task, 'metadata'):
        meta = env.task.metadata
        save_dir = os.path.join(save_dir_base, meta['subset'], meta['category'], meta['template'])
        output_path = os.path.join(save_dir, f"history_{idx}.json")
    else:
        save_dir = save_dir_base
        output_path = os.path.join(save_dir, f'history_{idx}.json')

    if os.path.exists(output_path):
        if getattr(args, 'resume', False):
            try:
                with open(output_path) as f:
                    _hist = json.load(f)
                _last = _hist[-1] if _hist else {}
                if isinstance(_last, dict) and 'result' in _last:
                    _score = float(_last.get('score', 0)) if _last.get('result') == 'success' else 0.0
                    env.close()
                    return _score
            except Exception:
                logger.warning(f"[Resume] Experiment {idx} file broken. Re-running...")
        else:
            env.close()
            return 0

    os.makedirs(save_dir, exist_ok=True)
    agent = EHRAgent(agent_config, permitted_actions=task_cls.permitted_actions)
    attempts = 0
    for step_idx in range(args.num_steps):
        action, params = agent.act(obs)
        while action == 'error':
            if getattr(agent, 'context_length_exceeded', False):
                logger.error(f"Task {args.task}-{idx} Failure: Context length exceeded, terminating immediately.")
                agent.conversation_history.append({'result': 'failure', 'context_length_exceeded': True, 'total_time': round(time.time() - experiment_start_time, 2)})
                save_conversation_history(agent.conversation_history, output_path)
                env.close()
                return 0
            n_retry = config['Agent']['n_retry']
            logger.error(f"Task {args.task}-{idx} Failure: Agent action failed for {n_retry} times.")
            attempts += 1
            if attempts >= config['Env']['n_retry']:
                agent.conversation_history.append({'result': 'failure', 'total_time': round(time.time() - experiment_start_time, 2)})
                success = 0
                output_path = os.path.join(save_dir, f'history_{idx}.json')
                save_conversation_history(agent.conversation_history, output_path)
                env.close()
                return success
            time.sleep(1)
            action, params = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action, **params)
        if done:
            break


    total_time = round(time.time() - experiment_start_time, 2)
    if done:
        agent.conversation_history.append({'result': 'success' if reward else 'failure', 'score': reward, 'total_time': total_time})
        success = reward
    else:
        # Retain the final tool reply even when no next model turn is allowed.
        agent.conversation_history.append({'role': 'user', 'content': obs.get('env_message') or '(No output)'})
        agent.conversation_history.append({'result': 'failure', 'score': 0, 'total_time': total_time})
        success = 0

    task_metadata = {
        "task_id": idx,
        "question": env.task.question,
    }
    # Retain source metadata when provided by the task.
    if hasattr(env.task, 'standard_sql'):
        task_metadata["standard_sql"] = env.task.standard_sql
    if hasattr(env.task, 'answer'):
        task_metadata["ground_truth"] = env.task.answer
    if hasattr(env.task, 'expected_result'):
        task_metadata["expected_result"] = env.task.expected_result
    if hasattr(env.task, 'metadata'):
        task_metadata["metadata"] = env.task.metadata

    # Insert metadata at the very beginning of the trajectory
    agent.conversation_history.insert(0, {"task_metadata": task_metadata})
    
    output_path = os.path.join(save_dir, f'history_{idx}.json')
    save_conversation_history(agent.conversation_history, output_path)
    env.close()
    return success

run_single_experiment_ray = ray.remote(run_single_experiment)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Update a progress bar when joblib batches complete."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def async_run_experiments(args, config, n_jobs, parallel_backend="ray"):
    if args.start_idx > args.end_idx:
        logging.warning("No experiments to run")
        return
    success_rate = 0
    try:
        if parallel_backend == 'joblib':
            from joblib import Parallel, delayed
            indices = list(range(args.start_idx, args.end_idx))
            existing_scores = {}

            if getattr(args, 'resume', False) and args.task == 'ehr_complex':
                save_dir = os.path.join(args.work_dir, args.task, args.result_dir_tag, args.mode)
                existing_line_idxs = set()
                for root, dirs, files in os.walk(save_dir):
                    for fname in files:
                        if not (fname.startswith('history_') and fname.endswith('.json')):
                            continue
                        fpath = os.path.join(root, fname)
                        if os.path.getsize(fpath) == 0:
                            continue
                        try:
                            with open(fpath) as _f:
                                _hist = json.load(_f)
                            _last = _hist[-1] if _hist else {}
                            if isinstance(_last, dict) and 'result' in _last:
                                try:
                                    line_idx = int(fname[len('history_'):-len('.json')])
                                    if args.start_idx <= line_idx < args.end_idx:
                                        existing_line_idxs.add(line_idx)
                                        existing_scores[line_idx] = float(_last.get('score', 0)) if _last.get('result') == 'success' else 0.0
                                except ValueError:
                                    pass
                        except Exception:
                            pass

                original_count = len(indices)
                indices = [idx for idx in indices if idx not in existing_line_idxs]
                skipped = original_count - len(indices)
                print(f"[Resume] Skipping {skipped} completed tasks; running {len(indices)} tasks")

            with tqdm_joblib(tqdm(desc="Total Experiment Progress", total=len(indices))) as progress_bar:
                results = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(run_single_experiment)(args, config, idx)
                    for idx in indices
                )
            total_count = len(existing_scores) + len(results)
            success_rate = (sum(existing_scores.values()) + sum(results)) / total_count if total_count else 0.0
            print(f'Success Rate: {success_rate}')
        elif parallel_backend == "ray":
            ray.init(num_cpus=n_jobs)
            indices = range(args.start_idx, args.end_idx)
            futures = [
                run_single_experiment_ray.remote(args, config, idx)
                for idx in indices
            ]
            results = ray.get(futures)
            success_rate = sum(results) / len(results)
            print(f'Success Rate: {success_rate}')
            ray.shutdown()
        else:
            raise ValueError(f"Unsupported parallel backend: {parallel_backend}")
    finally:
        logger.info('Experiment finished.')
        log_file = os.path.join(args.work_dir, "running_records.jsonl")
        model_name = config['Agent']['llm']['model_name']
        temperature = config['Agent']['llm']['temperature']
        with open(log_file, "a+") as f:
            record_line = (
                f"Task: {args.task} | "
                f"Model: {model_name} | "
                f"Temp: {temperature} | "
                f"Accuracy: {success_rate}\n"
            )
            f.write(record_line)

def main():
    # initialization
    args = parse_arguments()
    
    if args.config_path:
        config = load_config(args.config_path)
        
        config = update_dynamic_config(config, args)
        
        args = convert_config_to_args(config, args)

    if args.end_idx == -1:
        import json
        metadata_file = config['Data']['metadata_path']
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        if args.task in metadata:
            args.end_idx = metadata[args.task][args.mode]
        else:
            logging.warning(f"Task {args.task} not found in metadata.json, using default end_idx or logic.")

    credentials = load_credentials(args.credentials_path)
    set_environment_variables(credentials)
    
    
    env_config_tmp_dir = create_env_config_dir(args.work_dir, args.task, args.result_dir_tag)

    import yaml
    config_backup_path = os.path.join(env_config_tmp_dir, "run_config.yaml")
    with open(config_backup_path, 'w') as f:
        yaml.dump(config, f)

    if getattr(args, 'rescore', False):
        print(f"Rescoring saved answers")
        rescore_all_experiments(args, config)
        return

    if getattr(args, 'resume', False):
        save_dir = os.path.join(args.work_dir, args.task, args.result_dir_tag, args.mode)
        status = scan_workdir_status(save_dir, args.start_idx, args.end_idx, task=args.task)
        total = args.end_idx - args.start_idx
        print(f"Resuming evaluation")

        if args.task == 'ehr_complex':
            n_done   = status['completed']
            n_failed = status['failed']
            n_broken = status['broken']
            n_total_files = status['total_files']
            print(f"[Resume] Tasks: {total}; result files: {n_total_files}; completed: {n_done}; incomplete: {n_failed}; unreadable: {n_broken}")
        else:
            n_done    = len(status['completed'])
            n_failed  = len(status['failed'])
            n_missing = len(status['missing'])
            n_broken  = len(status['broken'])
            print(f"[Resume] Tasks: {total}; completed: {n_done}; incomplete: {n_failed}; missing: {n_missing}; unreadable: {n_broken}")
            if n_done > 0:
                avg_score = 0.0
                for idx in status['completed']:
                    try:
                        path = os.path.join(save_dir, f'history_{idx}.json')
                        with open(path) as f:
                            hist = json.load(f)
                        last = hist[-1] if hist else {}
                        avg_score += float(last.get('score', 0))
                    except Exception:
                        pass
                avg_score /= n_done
                print(f"[Resume] Mean score for completed tasks: {avg_score:.4f}")
            print(f"[Resume] Running {n_missing + n_failed + n_broken} remaining tasks")


    # run experiments
    if not args.async_run:
        sequential_run_experiments(args, config)
    else:
        async_run_experiments(args, config, args.n_jobs, args.parallel_backend)


if __name__ == '__main__':
    main()
