import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
import psutil
import time

import pandas as pd
from sklearn.model_selection import train_test_split

import random
from sklearn.metrics import accuracy_score ,precision_score, recall_score, f1_score ,roc_auc_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from Trainers import Trainer
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import csv
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

parent_path=os.path.dirname(os.getcwd())
file_path = parent_path +'/MLL.csv'

data = pd.read_csv(file_path, sep=',')

# 2. Encoding Categorical Variables:
class_mapping = {'ALL': 0, 'AML': 1, 'MLL': 2}
data['class'] = data['class'].map(class_mapping)

Y = data['class']
columns_to_exclude = ['class']
X = data.drop(columns=columns_to_exclude)
X.columns = range(len(X.columns))

# Apply SMOTE
smote = SMOTE(random_state=4)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X, Y)

# Scale the features
scaler = preprocessing.StandardScaler().fit(X_train_SMOTE)
X_train_SMOTE_scaled_array = scaler.transform(X_train_SMOTE)
X_train_SMOTE_scaled = pd.DataFrame(X_train_SMOTE_scaled_array, columns=X_train_SMOTE.columns)

#Data splitting
X_train, X_test, y_train,  y_test = train_test_split(X_train_SMOTE_scaled, y_train_SMOTE, test_size = 0.3)

#Number of features
F = len(X.columns)
print(F)

# Configuration paramaters for the whole setup
gamma = 0.99 # Discount factor for past rewards
epsilon = 0.99 # Epsilon greedy parameter
batch_size = 64  # Size of batch taken from replay buffer

max_steps_per_episode = 1000
trainers_steps = int(max_steps_per_episode/2)

num_actions = 2
state_size = 49
terminal = False
hyperparameter = 0.001
num_agents = F

# Experience replay buffers
action_history = []
agents = list(range(num_agents))
agent_lists = {agent_id: [] for agent_id in agents}
state_history = []
rewards_list = [[0, 0, 0 , 0, 0, 0, 0, 0, 0]]* max_steps_per_episode
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Maximum replay length
max_memory_length = 2000
update_after_actions = 10
# How often to update the target network
update_target_network = 1000
# Using huber loss for stability
loss_function = keras.losses.Huber()

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, clipnorm=1.0)

def show_RAM_usage():
    py = psutil.Process(os.getpid())
    print("**********************************************************")
    print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))

def create_q_model():
    x = Input(shape=(state_size,))

    # a series of fully connected layer for estimating Q(s,a)
    y1 = tf.keras.layers.Dense(64, activation='relu')(x)
    y2 = tf.keras.layers.Dense(8, activation='relu')(y1)
    z = tf.keras.layers.Dense(num_actions, activation="linear")(y2)
    model = keras.Model(inputs=x, outputs=z)
    model.compile(optimizer=optimizer)
    return model

def reset():  # initialize the world
    M = random.randint(1, F)
    initial_state = random.sample(range(F), M)
    return initial_state

def state_representation(St):
    x_train = X_train[St]
    if x_train.empty:
        return [0] * 49
    else : 
        S = x_train.values  
        D = pd.DataFrame(data=S).describe().T
        D_meta = D.describe().T
        D_meta_new = D_meta.loc[:, ['mean', 'std', 'min', '25%' , '50%', '75%', 'max']]
        D_meta_new = D_meta_new.drop(D_meta_new.index[0])
        s = D_meta_new.values.flatten()
        return s
def accuracy(input):
    x_train = X_train[input]
    x_test = X_test[input]
    information_gain = mutual_info_classif(x_train, y_train)
    sum = 0
    for _, ig in zip(input, information_gain):
        sum = sum + ig
    rf_classifier = XGBClassifier(objective="binary:logistic", random_state=42)
    rf_classifier.fit(x_train, y_train)    
    y_pred = rf_classifier.predict(x_test)
    accur = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    label_binarizer = LabelBinarizer()
    y_test_encoded = label_binarizer.fit_transform(y_test)
    y_pred_encoded = label_binarizer.transform(y_pred)
    aucScore = roc_auc_score(y_test_encoded, y_pred_encoded, multi_class='ovr')
    conf = confusion_matrix(y_test, y_pred)
    print(len(input), aucScore, accur)
    return aucScore ,accur, precision, recall, f1, list(conf), sum


def get_reward(features, prv_features, prv_auc):
    print('features' , len(features),'prv_features', len(prv_features))
    if len(features)==0:
        return 0 , 0, 0, 0, 0, 0, 0, 0   
    else: 
        auc, acc, precision, recall ,f1 , matrix, info_gain = accuracy(features)
        tot_f = len(features)
        tot_f_prv = len(prv_features)

        if auc > prv_auc and tot_f > tot_f_prv:
            R = auc - hyperparameter * info_gain
        elif auc > prv_auc and tot_f == tot_f_prv:
            R = auc
        elif auc > prv_auc and tot_f < tot_f_prv:
            R = auc
        
        elif auc == prv_auc and tot_f > tot_f_prv:
            R = auc - hyperparameter * info_gain
        elif auc == prv_auc and tot_f == tot_f_prv:
            R = auc   
        elif auc == prv_auc and tot_f < tot_f_prv:
            R = auc

        elif auc < prv_auc and tot_f > tot_f_prv:
            R = auc - hyperparameter * info_gain
        elif auc < prv_auc and tot_f  == tot_f_prv:
            R = auc - hyperparameter * info_gain
        elif auc < prv_auc and tot_f < tot_f_prv:
            R = auc - hyperparameter * info_gain
        R = R * 100
        return R , auc, acc, precision, recall, f1, matrix, tot_f   

def step(agents_actions, action_previous, previous_aucScore):        

    reward, auc_score, accuracyScore, precisionScore, recallScore, f1Score, conf_matrix ,feat_number = get_reward(agents_actions, action_previous,previous_aucScore)
    next_state = agents_actions
    next_state_vector = state_representation(next_state)
    if reward == 0:
        terminal = True
    else:
        terminal = False
    return [next_state, next_state_vector, reward, auc_score, accuracyScore , precisionScore, recallScore , f1Score, conf_matrix, feat_number,terminal]

def reset_previous_actions():     
    initial_previous_actions = [1] * num_agents
    return initial_previous_actions

def selected_features(list_actions):       
    features = []
    for i, act in enumerate(list_actions):
        if act == 1:
            features.append(i)
    return features

def clear_session():
    tf.keras.backend.clear_session()

models = [create_q_model() for _ in range(num_agents)]
models_target = [create_q_model() for _ in range(num_agents)]

start_time = time.time()
state = reset()
episode_reward = 0
previous_actions = [i for i in range(num_agents)]
previous_aucScore,_, _, _, _, _, _= accuracy(previous_actions)

for timestep in range(0, max_steps_per_episode):
    print("********** timestep **********", timestep)
    state = np.array(state)
    state = state.ravel()
    frame_count += 1
    state_vectors = state_representation(state)
    actions = []
    #  for actions list
    for i in range(num_agents):
        # For each agent, choose an action using epsilon-greedy
        if epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.convert_to_tensor(state_vectors)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = models[i](state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        agent_lists[i].append(action)
        actions.append(action)

    participated_agents = previous_actions                   
    initial_actions_feat = selected_features(actions)

    if timestep <= trainers_steps:        

        trainer = Trainer(participated_agents)
        selected_features1, k, assertive_agents,hesitant_agents= trainer.Warm_up(participated_agents, initial_actions_feat)
        f1_score_kbest, advice_k_best = trainer.k_best_score(participated_agents, int(k),X_train, X_test, y_train,  y_test)
        f1_score_DT, advice_DT = trainer.decision_tree_score_MultiClass(participated_agents, assertive_agents, hesitant_agents,X_train, X_test, y_train,  y_test)

        if f1_score_kbest > f1_score_DT:          
            for l in range(len(hesitant_agents)):
                if hesitant_agents[l] in advice_k_best:
                    selected_features1.append(hesitant_agents[l])
        else:
            for o in range(len(hesitant_agents)):
                if hesitant_agents[o] in advice_DT:
                    selected_features1.append(hesitant_agents[o])
        for i in range(len(assertive_agents)):
            selected_features1.append(assertive_agents[i])
        if len(selected_features1) == 0:
            selected_features1 = reset()

    else:
        selected_features1 = list(set(participated_agents).intersection(set(initial_actions_feat)))
        if len(selected_features1) == 0:
            selected_features1 = reset()

    # Apply the sampled action in the environment for each agent
    state_next, state_next_vector, reward, aucScore, accScore, precisionScore, recallScore, f1Score, matrix_conf, feat_numb, done = step(selected_features1, participated_agents, previous_aucScore)

    state_next = np.array(state_next).ravel()
    state_next_vector = np.array(state_next_vector).ravel()

    episode_reward += reward

    # Save actions and states in replay buffer
    state_history.append(state_vectors)
    state_next_history.append(state_next_vector)
    done_history.append(done)
    rewards_history.append(reward)

    rewards_list[timestep] = [accScore*100, reward, aucScore*100 , feat_numb, precisionScore*100, recallScore*100, f1Score*100, matrix_conf, selected_features1]
    excel_file_path1 = 'PredictionsMLL.txt'
    with open(excel_file_path1, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['AccuracyScore', 'Reward', 'AUCScore' , 'FeatureNumber', 'PrecisionScore', 'RecallScore', 'f1Score', 'Matrix_conf', 'selected_features1'])
        for values in rewards_list:
            csv_writer.writerow(values)

    state = state_next
    previous_actions = selected_features1
    previous_aucScore = aucScore    
    print("frame_count", frame_count)
    # Update every fourth frame and once batch size is over 32
    if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
        for agent_idx in range(num_agents):           
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            # Use the respective Q-network model for each agent
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [agent_lists[agent_idx][i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = models_target[agent_idx].predict(state_next_sample)

            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )           

             # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample          

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = models[agent_idx](state_sample)
                
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)
            # Backpropagation
            grads = tape.gradient(loss, models[agent_idx].trainable_variables)
            optimizer.apply_gradients(zip(grads, models[agent_idx].trainable_variables))

            if frame_count % update_target_network == 0:
                print("update the the target network with new weights")
                # update the the target network with new weights
                models_target[agent_idx].set_weights(models[agent_idx].get_weights())
            if len(agent_lists[agent_idx]) > max_memory_length:
                del agent_lists[agent_idx][:1]
        
        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del done_history[:1]
    clear_session()
    show_RAM_usage()

end_time = (time.time()-start_time)/60
print("time spent :",end_time)
