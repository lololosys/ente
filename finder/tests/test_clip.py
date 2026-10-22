from conftest import fixture_image

from finder.clip import clip_image, clip_text
from finder.clip_tokenizer import Tokenizer


def test_tokenizer_matches_openai_clip(models):
    t = Tokenizer(models.vocab_path)
    ids = t.tokenize("A photo of a dog!")
    assert ids.shape == (77,)
    assert list(ids[:8]) == [49406, 320, 1125, 539, 320, 1929, 256, 49407]
    assert ids[8:].sum() == 0


def test_tokenizer_truncates_long_text(models):
    ids = Tokenizer(models.vocab_path).tokenize("word " * 200)
    assert ids.shape == (77,) and ids[-1] == 49407


def test_clip_text_image_dot_ranks_matching_caption_higher(models):
    img = clip_image(models, fixture_image("person_a_1.jpg"))
    suit = clip_text(models, "a portrait of a man in a suit")
    cake = clip_text(models, "a wedding cake on a table")
    assert float(img @ suit) > float(img @ cake)
    assert float(img @ suit) > 0.175
