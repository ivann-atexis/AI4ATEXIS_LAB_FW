from common.apps.ocr import create_ocr_app as _create_ocr_app
from common.runtime import ModelSession
from common.types import AppOptions


def create_ocr_app(session: ModelSession, options: AppOptions):
    return _create_ocr_app(
        session,
        options,
        title="Xiaomi-OCR-0",
        description=(
            "OCR local avec Xiaomi-OCR-0 sous Transformers et PP-DocLayoutV3. "
            "Le mode regions transcrit les boites detectees ; le mode page "
            "transcrit l'image entiere. Cliquez sur une boite pour voir son texte."
        ),
        modes=("regions", "page"),
        default_mode="regions",
    )


def run(session: ModelSession, options: AppOptions) -> None:
    create_ocr_app(session, options).launch(
        server_name=options.host,
        server_port=options.port,
        share=False,
    )
