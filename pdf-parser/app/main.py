# main.py
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# from marker.convert import convert_single_pdf
# from marker.logger import configure_logging
# from marker.models import load_all_models
import logging

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.converters.pdf import PdfConverter
from marker.logger import configure_logging
from marker.models import create_model_dict
from marker.output import save_output

app = FastAPI()

# Configure logging
# configure_logging()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Service is starting, models are loading...")

# Load models globally, only once
# model_lst = load_all_models()
models = create_model_dict()
config_parser = ConfigParser({"langs": ["Japanese"], "output_format": "markdown"})
converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=models,
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer()
)

logger.info("Model loading completed.")

@app.post("/convert/")
async def convert_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")

    # Temporarily save the uploaded file
    try:
        with open("temp.pdf", "wb") as buffer:
            buffer.write(await file.read())

        logger.info("Starting PDF conversion process.")
        # Use the pre-loaded models
        # full_text, images, out_meta = convert_single_pdf("temp.pdf", model_lst, max_pages=None, langs=["Japanese"], batch_multiplier=1)
        rendered = converter("temp.pdf")
        print(rendered.markdown)
        logger.info("PDF conversion completed.")
        return JSONResponse(content={"text": rendered.markdown})
    except Exception as e:
        logger.error(f"Error occurred while processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
