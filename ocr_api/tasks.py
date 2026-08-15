import os
import shutil
import time
from datetime import datetime

import json
from celery import shared_task
from celery.utils.log import get_task_logger
from django.db import transaction
from django.conf import settings
from prometheus_client import Counter, Gauge

task_logger = get_task_logger(__name__)

# metrics
TASK_RUNS = Counter('active_training_runs_total', 'Total active training task runs')
TASK_SUCCESSES = Counter('active_training_success_total', 'Successful active training runs')
TASK_FAILURES = Counter('active_training_failure_total', 'Failed active training runs')
LAST_BASELINE_CER = Gauge('active_training_last_baseline_cer', 'Last baseline CER')
LAST_NEW_CER = Gauge('active_training_last_new_cer', 'Last new CER')
from django.utils import timezone

from .models import EditedOCRExample
from . import ocr_engine
import torch

# default weights directory configured in ocr_engine
WEIGHTS_DIR = getattr(ocr_engine, 'WEIGHTS_PATH', None)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_active_training(self, threshold=10):
    TASK_RUNS.inc()
    # Prefer a Redis-backed distributed lock when Redis is available (recommended for multi-host).
    # Fallback to a simple file lock for single-host / Windows dev environments.
    redis_lock = None
    use_redis_lock = False
    lock_path = os.path.join(getattr(ocr_engine, 'BASE_DIR', '.'), '.run_active_training.lock')

    # ------------------
    # Redis lock attempt
    # ------------------
    try:
        import redis as _redis
        redis_url = os.environ.get('REDIS_URL') or os.environ.get('CELERY_BROKER_URL') or getattr(settings, 'CELERY_BROKER_URL', None)
        if redis_url:
            try:
                redis_client = _redis.from_url(redis_url)
                lock_key = 'backend_ocr:run_active_training_lock'
                # 6 hours TTL to match file-lock behaviour
                redis_lock = redis_client.lock(lock_key, timeout=6 * 3600)
                acquired = redis_lock.acquire(blocking=False)
                if acquired:
                    use_redis_lock = True
                    task_logger.info('Acquired Redis lock for active training')
                else:
                    task_logger.info('Another active training run holds the Redis lock; skipping.')
                    return {'status': 'skipped', 'reason': 'redis_lock_present'}
            except Exception as _e:
                task_logger.debug('Redis lock attempt failed: %s', _e)
    except Exception:
        # redis client not available or import failed — fallback to file lock
        pass

    # ------------------
    # File lock fallback (single-host / Windows)
    # ------------------
    if not use_redis_lock:
        if os.path.exists(lock_path):
            try:
                with open(lock_path, 'r') as fh:
                    data = json.load(fh)
                ts = data.get('ts')
                if ts and (time.time() - ts) < 6 * 3600:
                    task_logger.info('Another active training run is in progress or recently completed; skipping.')
                    return {'status': 'skipped', 'reason': 'lock present'}
            except Exception:
                pass
        try:
            with open(lock_path, 'w') as fh:
                json.dump({'ts': time.time(), 'task_id': self.request.id}, fh)
        except Exception:
            task_logger.warning('Could not create lock file; proceeding anyway')

    pending_qs = EditedOCRExample.objects.filter(used=False).order_by('created_at')
    total_pending = pending_qs.count()

    if total_pending < threshold:
        return {'status': 'skipped', 'reason': f'not enough edits ({total_pending})'}

    examples = []
    for edit in pending_qs:
        record = edit.ocr_record
        image_path = None
        try:
            image_path = record.image.path if record.image and hasattr(record.image, 'path') else None
        except Exception:
            image_path = None

        examples.append({
            'ocr_record_id': record.id,
            'original_text': record.extracted_text or '',
            'edited_text': edit.edited_text or '',
            'image_path': image_path,
            'user_id': edit.user_id,
        })

    backup_dir = None
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    try:
        if WEIGHTS_DIR and os.path.exists(WEIGHTS_DIR):
            backup_dir = f'{WEIGHTS_DIR}_backup_{timestamp}'
            shutil.copytree(WEIGHTS_DIR, backup_dir)

        task_logger.info('Starting active training with %d examples', len(examples))
        result = ocr_engine.active_train_on_examples(
            examples,
            base_model_dir=ocr_engine.BASE_MODEL_NAME,
            adapter_dir=ocr_engine.WEIGHTS_PATH,
            output_root=os.path.join(ocr_engine.BASE_DIR, 'weights_active_runs'),
            num_epochs=2,
            batch_size=4,
            learning_rate=1e-5,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        # support both boolean and dict return for backward compatibility
        success = False
        new_weights_dir = None
        if isinstance(result, dict):
            success = result.get('success', False)
            new_weights_dir = result.get('new_weights_dir')
        else:
            success = bool(result)

        if success:
            # if new weights are produced, replace WEIGHTS_DIR safely
            if new_weights_dir:
                try:
                    if WEIGHTS_DIR and os.path.exists(WEIGHTS_DIR):
                        shutil.rmtree(WEIGHTS_DIR)
                    shutil.copytree(new_weights_dir, WEIGHTS_DIR)
                except Exception:
                    # failed to promote new weights: attempt rollback
                    if backup_dir and os.path.exists(backup_dir):
                        if WEIGHTS_DIR and os.path.exists(WEIGHTS_DIR):
                            shutil.rmtree(WEIGHTS_DIR)
                        shutil.copytree(backup_dir, WEIGHTS_DIR)
            now = timezone.now()
            pending_qs.update(used=True, used_at=now)
            TASK_SUCCESSES.inc()
            try:
                if isinstance(result, dict):
                    baseline = result.get('baseline_cer')
                    new = result.get('new_cer')
                    if baseline is not None:
                        LAST_BASELINE_CER.set(float(baseline))
                    if new is not None:
                        LAST_NEW_CER.set(float(new))
            except Exception:
                pass
            task_logger.info('Active training completed successfully')
            return {'status': 'success', 'examples_trained': len(examples)}
        else:
            # training failed: restore backup
            if backup_dir and os.path.exists(backup_dir):
                try:
                    if WEIGHTS_DIR and os.path.exists(WEIGHTS_DIR):
                        shutil.rmtree(WEIGHTS_DIR)
                    shutil.copytree(backup_dir, WEIGHTS_DIR)
                except Exception:
                    pass
            TASK_FAILURES.inc()
            task_logger.warning('Active training run did not promote weights')
            return {'status': 'failed'}
    except Exception as exc:
        # on unexpected failure try to restore
        if backup_dir and os.path.exists(backup_dir):
            try:
                if WEIGHTS_DIR and os.path.exists(WEIGHTS_DIR):
                    shutil.rmtree(WEIGHTS_DIR)
                shutil.copytree(backup_dir, WEIGHTS_DIR)
            except Exception:
                pass
        TASK_FAILURES.inc()
        task_logger.exception('Active training error: %s', exc)
        return {'status': 'error', 'error': str(exc)}
    finally:
                try:
                        if use_redis_lock and redis_lock is not None:
                                try:
                                        redis_lock.release()
                                        task_logger.info('Released Redis lock for active training')
                                except Exception:
                                        task_logger.warning('Failed to release Redis lock')
                        else:
                                if os.path.exists(lock_path):
                                        os.remove(lock_path)
                except Exception:
                        pass


@shared_task(bind=True)
def process_ocr_record(self, record_id):
    """Process a single OCRRecord in the background (moved from views to Celery task)."""
    logger = get_task_logger(__name__)
    try:
        from .models import OCRRecord
        record = OCRRecord.objects.filter(pk=record_id).first()
        if not record:
            logger.warning('process_ocr_record: record not found %s', record_id)
            return {'status': 'not_found'}

        record.status = OCRRecord.STATUS_PROCESSING
        record.error_message = None
        record.save(update_fields=['status', 'error_message'])

        # default to CPU for inference to avoid persistent GPU allocation and OOMs in shared environments
        device = os.environ.get('OCR_DEVICE', 'cpu')
        engine = ocr_engine.get_ocr_engine(device=device)
        try:
            extracted_text = (engine.predict(record.image.path) or '').strip()
            logger.info("Full-page OCR text for record_id=%s: '%s'", record.id, extracted_text)
        except Exception:
            logger.exception('Full-page OCR failed for record_id=%s', record.id)
            extracted_text = ''
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass

        record.extracted_text = extracted_text
        record.status = OCRRecord.STATUS_COMPLETED
        record.error_message = None
        record.save(update_fields=['extracted_text', 'status', 'error_message'])
        return {'status': 'success'}
    except Exception as exc:
        try:
            from .models import OCRRecord
            record = OCRRecord.objects.filter(pk=record_id).first()
            if record:
                record.error_message = str(exc)
                record.status = OCRRecord.STATUS_FAILED
                record.save(update_fields=['status', 'error_message'])
        except Exception:
            pass
        logger.exception('Unhandled OCR background exception for record_id=%s', record_id)
        return {'status': 'error', 'error': str(exc)}


    
