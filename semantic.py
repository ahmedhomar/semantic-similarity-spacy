import spacy


# Load the installed model "en_core_web_md"
nlp = spacy.load("en_core_web_md")

# Create a Doc objects
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Generate the tokens
tokens = nlp("cat apple monkey banana yarn core ")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana",
    "Why is my car full of caterpillars",
]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(f"{sentence} - , {similarity}")

    """
NOTES:
0.8 - 1.0 = very similar
0.6 - 0.8 = similar

* Apple and banana most similar: both fruits
* Cat and monkey = 0.59 => both animals, mammals
* Apple : core = 0.224 but banana : core = 0.06.
* Apples have a core but bananas don't.
* Least similar was cat : core
* Monkey : banana = 0.404 => monkeys eat bananas.


cat apple 0.2036806046962738
cat monkey 0.5929930210113525
cat banana 0.2235882580280304
cat yarn 0.19720639288425446
cat core -0.13286946713924408
apple monkey 0.2342509925365448
apple banana 0.6646699905395508
apple yarn 0.1824081838130951
apple core 0.22442923486232758
monkey banana 0.4041501581668854
monkey yarn 0.17825400829315186
monkey core -0.023859359323978424
banana yarn 0.27414757013320923
banana core 0.060533925890922546
yarn core 0.11178915202617645

Order of similarity to "Why is my cat on the car" (descending):

Hello, there is my car - , 0.8033180111627156
Why is my car full of caterpillars - , 0.802785247279939
I've lost my car in my car - , 0.6787541571030323
I will name my dog Diana - , 0.6491444739190607
where did my dog go - , 0.630065230699739
I'd like my boat back - , 0.5624940517078084


--- en_core_web_md versus en_core_web_sm ---

* The similarity values for the larger model (en_core_web_md) are much higher
than those of the smaller model (en_core_web_sm)

* On running example.py using en_core_web_sm we get the following warning:

The model you're using has no word vectors loaded,
so the result of the Doc.similarity method will be based on the tagger,
parser and NER, which may not give useful similarity judgements.
This may happen if you're using one of the small models, e.g. `en_core_web_sm`,
which don't ship with word vectors and only use context-sensitive tensors.
You can always add your own word vectors,
or use one of the larger models instead if available.

    """
