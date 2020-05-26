# Common file for stmweb/nets and external networks' modules. Is stored in stmweb and linked to other folders

import codecs
import os
import pickle
import sys
import time

# print('Pickle protocols: ', pickle.HIGHEST_PROTOCOL, pickle.DEFAULT_PROTOCOL)

# Serializes any object into printable string
def encodeToStr(obj):
    # print('Encoding %s (type %s)' % (str(obj), str(type(obj))))
    encData = codecs.encode(pickle.dumps(obj, protocol=3), 'base64').decode()
    encData = encData.replace('\n', '')
    # print('Encoded: %s' % (str(encData)))
    return encData

def decodeObject(str):
    return pickle.loads(codecs.decode(str.encode(), 'base64'))

# Cuts too long keys and values in order not to spoil display/log space
def cutLongImageInfo(imageInfo):
    info = {}
    if not imageInfo or not isinstance(imageInfo, dict):
        return imageInfo
    for key, val in imageInfo.items():
        val = str(val)
        if len(val) > 30:
            val = val[:30] + '...'
        if len(key) > 30:
            key = key[:30] + '...'
        info[key] = val
    return info

def cutLongStrings(data):
    if isinstance(data, list):
        # print('Processing list ', data)
        result = []
        for val in data:
            result.append(cutLongStrings(val))
    elif isinstance(data, dict):
        result = {}
        for key, val in data.items():
            # print('Key %s: %s' % (key, str(type(val))))
            result[cutLongStrings(key)] = cutLongStrings(val)
    else:
        # print('Processing str ', data)
        val = str(data)
        if len(val) > 100:
            val = val[:100] + '...'
        return val
    return result


# Opens existing or creates log file
def openLogFile(filePath):
    import logging

    logParams = dict(level=logging.DEBUG,
                      format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
    # logParams['filename'] = filePath
    # logParams['filemode'] = 'a'

    logging.basicConfig(**logParams)
    logger = logging.getLogger('stm_' + os.path.basename(filePath))

    if logger.handlers:
        return logger    # Already initialized

    file_handler = logging.FileHandler(filePath, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        # Here asctime already contains milliseconds for some reason
    logger.addHandler(file_handler)

    # logger.info('Processing started')
    # logger.info('')
    # logger.info('-' * 50)
    # logger.info('Processing started. Process id %d, options: ' % os.getpid())
    # for attr, value in CONFIG.items():
    #     logger.info('  %s = %s' % (attr, value))

    return logger
    # global LOG
    # LOG = logger


# This can add execution time between each standard print method calls
builtinPrint = print

def print_timeMeasure(*args):
    try:
        t = time.clock()
        builtinPrint("%5.3f (%9.2f) " % (t, (t - print_timeMeasure.prevTime) * 1000000), end='')
        builtinPrint(*args)
        # (%.3f s)%s" % \
        #                   (iterNum, math.sqrt(avgSqDiff), avgSqDiff, time2 - time0, ad
        # time0 = time2
        sys.stdout.flush()
        print_timeMeasure.prevTime = t
    except:
        builtinPrint("seconds from start (passed microseconds)")
        builtinPrint(*args)
        print_timeMeasure.prevTime = time.clock()

# print = print_timeMeasure