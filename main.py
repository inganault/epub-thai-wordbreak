#!/usr/bin/env python3
import onnxruntime as ort
import zipfile
import argparse
from pathlib import Path
import threading
import queue
from contextlib import contextmanager

ort_sess = ort.InferenceSession('sertiscorp-thai-word-segmentation.onnx')

thai_chars = {chr(x) for x in list(range(0x0E01, 0x0E3A)) + list(range(0x0E3F, 0x0E4D))}

# Construct lookup table to convert characters to input labels
_char_set = [chr(x) for x in [0] + [0x000A] + list(range(0x0020, 0x007F)) + \
        list(range(0x0E01, 0x0E3A)) + list(range(0x0E3F, 0x0E4D)) + list(range(0x0E50, 0x0E5A))]
_dictionary = {v:k for k,v in enumerate(_char_set)}

def break_word(chunk):
    inputs = [[_dictionary[ch] for ch in chunk]]
    lengths = [len(chunk)]
    outputs = ort_sess.run(None, {'inputs': inputs, 'lengths': lengths, 'training': [False]})
    out = []
    start = 0
    for idx, split in enumerate(outputs[0]):
        if split:
            out.append(chunk[start:idx])
            start = idx
    if start != lengths[0]:
        out.append(chunk[start:])
    return out[1:]

def scan_thai_chunk(text):
    chunk_type = None
    chunk_cur = ''
    for ch in text:
        is_thai = ch in thai_chars
        if is_thai == chunk_type:
            chunk_cur += ch
        else:
            if chunk_cur:
                yield (chunk_type, chunk_cur)
            chunk_cur = ch
            chunk_type = is_thai
    if chunk_cur:
        yield (chunk_type, chunk_cur)

def add_wbr(text):
    out = []
    for is_thai, chunk in scan_thai_chunk(text):
        if not is_thai:
            out += [chunk]
        else:
            words = break_word(chunk)
            out += ['&#8203;'.join(words)]
    return ''.join(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='epub-thai-wordbreak',
                        description='Add wordbreak to thai text')
    parser.add_argument('input', metavar='input.epub')
    parser.add_argument('-o', '--output')
    parser.add_argument('-w', '--worker', type=int, default=4, help='worker count')
    args = parser.parse_args()

    args.input = Path(args.input)
    if not args.input.exists():
        print("Input file doesn't exists")

    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + '_wbr')


    target_ext = {'htm', 'xhtml', 'html'}

    inp_q = queue.Queue(args.worker)
    output_q = queue.Queue()
    def worker():
        while True:
            order, zinfo, content = inp_q.get()
            if order is None:
                return
            content = content.decode('utf-8')
            content = add_wbr(content)
            content = content.encode('utf-8')
            output_q.put((order, zinfo, content))

    def output_worker():
        reorder = {}
        cur = 0
        with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as outzip:
            while True:
                order, zinfo, content = output_q.get()
                if order is None:
                    return
                reorder[order] = (zinfo, content)
                while cur in reorder:
                    zinfo, content = reorder[cur]
                    if zinfo is None:
                        return
                    outzip.writestr(zinfo, content)
                    cur += 1

    def input_worker():
        order = 0
        with zipfile.ZipFile(args.input) as inzip:
            for zinfo in inzip.infolist():
                print('Processing', zinfo.filename)
                is_target = zinfo.filename.split('.')[-1] in target_ext
                content = inzip.read(zinfo)
                if is_target:
                    inp_q.put((order, zinfo, content))
                else:
                    output_q.put((order, zinfo, content))
                order += 1
        output_q.put((order, None, None))

    threads = []
    try:
        for _ in range(args.worker):
            threads.append(threading.Thread(target=worker))
            threads[-1].start()
        threads.append(threading.Thread(target=output_worker))
        threads[-1].start()

        # Issue command
        input_worker()
    except:
        output_q.put((None, None, None))
        raise
    finally:
        # Wait thread
        for _ in range(args.worker):
            inp_q.put((None, None, None))
        for t in threads:
            t.join()
    print('ok')
