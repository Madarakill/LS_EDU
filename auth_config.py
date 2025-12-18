from __future__ import annotations

import os

# Роли:
# - user: загрузка CSV -> скачать predictions.csv (порог фиксированный 0.5)
# - operator: можно менять порог прогнозирования
# - admin: operator + админ-страница (пересобрать artifacts, заменить веса модели, переобучить модель)
#
# Пароли хранятся в виде хеша

# Ключ для сессий
SECRET_KEY = os.environ.get("LS_SECRET_KEY", "f8ad08a3e319c9c4a08e4708ebad54a6571dcee89affde205a991387d9a03b9e")

USERS = {
    # роль: user, пароль: 123qweASD
    "user": {
        "role": "user",
        "password_hash": "scrypt:32768:8:1$GCw4643xvn73EBvE$9c7bdff1e21dda27db2e9071f746733ade3dc749b56785b0f9516856283d950fcd5f4830cff2d0796dffd343625b5668fc6a569c480f751c2cd7ea08df6e82a3",
    },
    # роль: operator, пароль: 123qweASD!
    "operator": {
        "role": "operator",
        "password_hash": "scrypt:32768:8:1$GXPpYcAvlb1aavKv$4534798733c4aeb8d11355c1923a6762487d920a0bf36803121b77d9a7285170152df0b791bd5d06b99259b33f1019697578053611710cc04699c80115d99551",
    },
    # роль: admin, пароль: 123qweASD!@
    "admin": {
        "role": "admin",
        "password_hash": "scrypt:32768:8:1$FKJ5YzLpdzW1E1LX$4a6f657c667c6d82cde2f3da4dc6079df8b145437667fa5390b9243af1a586e4422c38587f3093a15734d5ce4f6a57b120b5a5cee0999bf2e08763887bf5b42c",
    },
}

