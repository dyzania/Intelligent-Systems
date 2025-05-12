import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # for POS tagging
nltk.download('maxent_ne_chunker')  # for named entity recognition
nltk.download('words')  # word lists needed by the chunker

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Part 2: Mystery Story
story = """Detective Verstappen had seen a lot in his twenty years with the CIA, 
but the case in Paris was unlike anything he’d encountered before. 
A priceless treasure had vanished from the Museum. 
under impossible circumstances no broken windows, no tripped alarms. 
The only clue was a single red rose left on the empty wall where the painting once hung. 
His contact at Interpol insisted it was the work of the elusive TAUGAMMA, 
an art theft syndicate rumored to operate across Europe. 
But as he examined the rose more closely, He noticed something odd,
a tiny engraving inside the stem, resembling a musical note. He'd seen that symbol 
before in a file buried deep within the FBI archives, linked not to thieves, but to spies."""

# Part 3: Tokenize and POS Tag
tokens = word_tokenize(story)
tagged = pos_tag(tokens)

print("Part-of-Speech Tagged Words:")
print(tagged)

# Part 4: Named Entity Recognition
ner_tree = ne_chunk(tagged)


print("\nNamed Entities:")
print(ner_tree)

# Part 5: Analyze and Report
people = []
locations = []
organizations = []

for subtree in ner_tree:
    if isinstance(subtree, Tree):
        POS = subtree.label()
        entity = " ".join([token for token, pos in subtree.leaves()])
        if POS == 'PERSON': #if words have POS Tagged (PERSON)
            people.append(entity) # add to peoples list
        elif POS == 'GPE' or POS == 'LOCATION': #if words have POS Tagged (GPE or LOCATION)
            locations.append(entity) # add to locations list
        elif POS == 'ORGANIZATION': #if words have POS Tagged (ORGANIZATION)
            organizations.append(entity) # add to organizations list

# Detective Case File Report    
print("\n***** DETECTIVE CASE FILE *****")
print("\nOriginal Story:\n")
print(story)

print("\nDetected Entities:")
print(f"People: {people}")
print(f"Locations: {locations}")
print(f"Organizations: {organizations}")

