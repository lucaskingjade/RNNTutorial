from preprocessingwq import *
def generate_sentence(model):
    #we start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    #Repeat util we get an end token
    #i=0
    while (not new_sentence[-1] == word_to_index[sentence_end_token]) and len(new_sentence)<50:
        #i =i+1
        #print "The %d th word is prediced" %i
        sys.stdout.flush()
        next_word_probs,s = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        #We don't want to sample unknown words
        #print "222The %d th word is prediced" %i
        sys.stdout.flush()
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(10,next_word_probs[-1])
            sampled_word = np.argmax(samples)
        #print "333The %d th word is prediced" %i
        sys.stdout.flush()
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

model = RNNNumpy(vocabulary_size,hidden_dim=50)
#model = RNNTheano(vocabulary_size, hidden_dim=50)
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
# save_model_parameters_theano('./data/trained-model-theano.npz', model)
load_model_parameters_theano('./data/trained-model-theano.npz', model)
#losses = traing_with_sgd(model,X_train[:100],Y_train[:100],learning_rate=0.01,nepoch=30,evaluate_loss_after=1)
#print "finish training"

num_sentences = 50
senten_min_length=10

for i in range(num_sentences):
    print "number sentence :%d" % i
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)



