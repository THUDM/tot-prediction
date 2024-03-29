import json
from os.path import join
import joblib

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def joblib_dump_obj(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    joblib.dump(obj, join(wfdir, wfname))
    logger.info('%s dumped.', wfname)


def joblib_load_obj(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    obj = joblib.load(join(rfdir, rfname))
    logger.info('%s loaded', rfname)
    return obj
    