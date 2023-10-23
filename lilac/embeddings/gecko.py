"""PaLM2 Gecko's embeddings."""

from typing import ClassVar, Iterable, cast
import lilac as ll
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import override
from ..env import env
from ..splitters.chunk_splitter import split_text

PALM_BATCH_SIZE = 1
NUM_PARALLEL_REQUESTS = 3
EMBEDDING_MODEL = 'textembedding-gecko@001'
EMBEDDING_SIZE = 768


class PaLMGecko(ll.TextEmbeddingSignal):
    """Computes embeddings using PaLM's gecko embedding API.

    <br>**Important**: This will send data to an external server!

    <br>To use this signal, you must get a GCP_PROJECT_ID env variable
    """
    name: ClassVar[str] = 'palm2-gecko'
    display_name: ClassVar[str] = 'PaLM2 Embeddings'

    @override
    def setup(self):
        gcp_project_id = env('GCP_PROJECT_ID')
        if not gcp_project_id:
            raise ValueError('`GCP_PROJECT_ID` environment variable not set.')
        try:
            from google.cloud import aiplatform
            from vertexai.language_models import TextEmbeddingModel
            aiplatform.init(project=gcp_project_id)
            self._embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        except ImportError:
            raise ImportError('Could not import the "vertex" python package. '
                              'Please install it with `pip install google-cloud-aiplatform`.')

    @override
    def compute(self, docs: Iterable[ll.RichData]) -> Iterable[ll.Item]:
        """Compute embeddings for the given documents."""

        @retry(wait=wait_fixed(5), stop=stop_after_attempt(15))
        def embed_fn(texts: list[str])-> list[np.ndarray]:
            """Compute embeddings from a list of texts.

            Args:
              list_of_texts: List of text to embed

            Returns
              np.array
                  2D array of shape (len(list_of_texts), embedding_size)
            """
            assert len(texts) == 1, 'PaLM API only supports batch size 1.'
            response = self._embedding_model.get_embeddings([texts[0]])[0].values
            return [np.array(response, dtype=np.float32)]

        docs = cast(Iterable[str], docs)
        split_fn = split_text if self._split else None

        yield from ll.compute_split_embeddings(
            docs=docs,
            batch_size=PALM_BATCH_SIZE,
            embed_fn=embed_fn,
            # Use the lilac chunk splitter.
            split_fn=split_fn,
            # How many batches to request as a single unit.
            num_parallel_requests=NUM_PARALLEL_REQUESTS)
