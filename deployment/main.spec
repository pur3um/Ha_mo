# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import importlib
sys.setrecursionlimit(5000)

block_cipher = None

def get_mediapipe_path():
    import mediapipe
    mediapipe_path = mediapipe.__path__[0]
    return mediapipe_path

added_files = [
    ('images/*.PNG','images'),
    ('model/VGGColab-2021-07-06_08-26-05.h5', 'model'),
    (os.path.join(os.path.dirname(importlib.import_module('tensorflow').__file__),"lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so"),"tensorflow/lite/experimental/microfrontend/python/ops/")
]

a = Analysis(['main.py'],
             pathex=['C:\\Users\\mein0\\01_playdata_final_project\\deployment'],
             binaries=[],
             datas=added_files,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

mediapipe_tree = Tree(get_mediapipe_path(), prefix='mediapipe', excludes=["*.pyc"])
a.datas += mediapipe_tree
a.binaries = filter(lambda x: 'mediapipe' not in x[0], a.binaries)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
