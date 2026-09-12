import torch
from sentence_transformers import SentenceTransformer
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')   # 22M params, frozen, CPU-fast
# embeds :: (L_sentences, 384) — L2-normalized, unit scale, ready to use

def embed_prompts(prompts):
    # prompts :: list[str]
    return torch.tensor(text_encoder.encode(prompts))   # (L, 384)

TEMPLATE = {
    0: ["a photo of an airplane in the sky",
        "an airplane flying over clouds",
        "a silver passenger plane on a runway",
        "a small propeller airplane",
        "a fighter jet in flight",
        "an airplane seen from below"],
    1: ["a photo of a car on a road",
        "a red sports car",
        "an old pickup truck parked",
        "a car driving on a highway",
        "a blue sedan in traffic",
        "a race car on a track"],
    2: ["a photo of a bird perched on a branch",
        "a small yellow bird in flight",
        "a blue jay on a tree branch",
        "a parrot with colorful feathers",
        "a seagull flying over the water",
        "a tiny sparrow on the ground"],
    3: ["a photo of a cat sitting on a windowsill",
        "a fluffy orange tabby cat",
        "a black cat with green eyes",
        "a gray kitten playing with yarn",
        "a cat sleeping curled up",
        "a white cat looking at the camera"],
    4: ["a photo of a deer in a forest",
        "a young deer standing in tall grass",
        "a deer with large antlers",
        "a brown deer looking at the camera",
        "a doe and her fawn in a meadow",
        "a deer drinking from a stream"],
    5: ["a photo of a dog running on grass",
        "a golden retriever fetching a ball",
        "a small brown puppy",
        "a black and white dog with a stick",
        "a German shepherd standing alert",
        "a dog in a park with its tongue out"],
    6: ["a photo of a green frog on a leaf",
        "a small tree frog on a branch",
        "a brown frog sitting on a rock",
        "a frog near a pond",
        "a bright green frog with big eyes",
        "a toad in the garden"],
    7: ["a photo of a horse in a field",
        "a brown horse galloping",
        "a white horse in a pasture",
        "a horse with a flowing mane",
        "a chestnut mare with a foal",
        "a black horse at a fence"],
    8: ["a photo of a ship on the ocean",
        "a large cargo ship at sea",
        "a sailboat with white sails",
        "a small motorboat on a lake",
        "a ship seen from the shore",
        "a fishing boat in the harbor"],
    9: ["a photo of a truck on a highway",
        "a large yellow semi truck",
        "a red pickup truck parked",
        "a delivery truck on a city street",
        "a dump truck at a construction site",
        "a white van on the road"],
}

def make_prompt(label):
    return TEMPLATE[int(label)][torch.randint(0, 6, (1,)).item()]

CLASS_EMB = torch.stack([embed_prompts(TEMPLATE[c]) for c in range(10)])
torch.save({'class_emb': CLASS_EMB}, 'txt_cache.pt')
