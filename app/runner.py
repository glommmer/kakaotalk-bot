import uvicorn
import logging.config
import sys


# healthcheck 로그를 필터링하는 최종 클래스
class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # 디버그 로그를 통해 확인된 uvicorn access 로그의 원시 데이터 구조:
        # record.args = (client_addr, method, path, http_version, status_code)
        # 예: ('10.0.0.142:0', 'GET', '/healthcheck', '1.0', 200)

        # 이 구조에 맞춰 경로와 상태 코드를 직접 확인합니다.
        if isinstance(record.args, tuple) and len(record.args) == 5:
            path = record.args[2]  # 3번째 요소: '/healthcheck'
            status_code = record.args[4]  # 5번째 요소: 200

            if path == "/healthcheck" and status_code == 200:
                return False  # healthcheck의 200 응답이면 로그를 남기지 않음

        return True  # 그 외 모든 경우는 로그를 남김


# 사용할 로깅 설정 (Python Dictionary 형태) - 이 부분은 수정 없음
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "healthcheck_filter": {
            "()": HealthCheckFilter,
        }
    },
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "filters": ["healthcheck_filter"],
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="127.0.0.1", port=8000, log_config=LOGGING_CONFIG, reload=True
    )
