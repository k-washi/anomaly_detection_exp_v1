# ml-exp-ad

異常検知の実験

# 実行環境作成(エディターモード)

`src.util.~`などモジュールのimportを行うために必要です。

```
pip install -e .
```

# test

テストの実行方法です。`pytest`か`tox`の使用方法を記載しています。

```
python -m pytest
```

toxで使用されるモジュールは、まず環境の作成を行います。

```
python -m tox
```


もし、toxの環境を作り直す

```
python -m tox -r
```

テストの実行方法です。
```
python -m tox -e py39
```

リンターによるチェックです。
```
python -m tox -e lint
```

# Docker

 CUDAによりpytorchのインストール方法が異なるので、適宜[公式](https://pytorch.org/)を参照し、インストールしてください。

```
docker-compose -f docker-compose-cpu.yml up -d
```

でコンテナを作成し、VS Codeの`ms-vscode-remote.remote-containers`から開発環境に入る

# gpu周り

もし、`docker-compose-gpu.yml`における以下の設定で上手くいかない場合

```
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            capabilities: [gpu]
```

以下に変更する。

```
runtime: nvidia
```

# vscode extensionの設定

1. view/command palletを開き、shellからcodeをインストール
2. 新しいshellを開く
3. 以下のコマンド実行 (権限は与えておく)

```
./.devcontainer/vscode_extentions_install_batch.sh
```