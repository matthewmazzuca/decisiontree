import numpy as np
import urllib

import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree
import os

class Learning:

	def __init__(self,training):
		self.training = training
		self.X, self.y, self.target = self.init_data()
		self.model = self.get_model()
		self.tree = self.tree_structure()
		self.num_nodes = self.num_nodes()
		self.children_right, self.children_left = self.children()
		self.feature = self.feature()
		self.threshold = self.threshold()

	def init_data(self):
		d2 = np.loadtxt(self.training, dtype=str, delimiter="\t")

		# print d2[1]
		y = d2[:,0]
		X = d2[:,1:,]
		target = X[0]
		# X.pop(0)
		# for i in range(len(X)):
		# 	for j in range(len(X[i])):
		# 		X[i][j] = int(X[i][j])

		for i in range(len(X)):
			print X[i]

		target = X[0]
		# self.clean_target(target)

		X = np.delete(X, 0, 0)

		y = np.delete(y, 0)
		X = X.astype(float)

		

		
		# print y
		# print X2[1]
		# print Y2[1]
		return X, y, target

	# def clean_target(self,target):
	# 	for item in target:
	# 		if '"' in item:
	# 			rows = item.split('"')
	# 			target[item] = rows[1]
	# 			# print target[item]

	def get_model(self):
		model = DecisionTreeClassifier(criterion='gini', presort=True)
		model.fit(self.X, self.y)
		return model

	def num_features(self):
		return self.model.n_features_

	# tree properties
	def tree_structure(self):
		return self.model.tree_

	def num_nodes(self):
		return self.tree.node_count

	def predict(self, guess):
		return self.model.predict(guess)

	def children(self):
		children_left = self.tree.children_left
		children_right = self.tree.children_right
		return children_right, children_left

	def feature(self):
		return self.tree.feature

	def predict_proba(self, fit):
		return self.model.predict_proba(fit)

	def score(self, fit, guess):
		# guess = self.target
		return self.model.score(fit, guess)

	def threshold(self):
		return self.tree.threshold

	def tree_print(self, *args, **kwargs):
		node_depth = np.zeros(shape=self.num_nodes)
		is_leaves = np.zeros(shape=self.num_nodes, dtype=bool)
		stack = [(0, -1)]  # seed is the root node id and its parent depth
		while len(stack) > 0:
		    node_id, parent_depth = stack.pop()
		    node_depth[node_id] = parent_depth + 1

		    # If we have a test node
		    if (self.children_left[node_id] != self.children_right[node_id]):
		        stack.append((self.children_left[node_id], parent_depth + 1))
		        stack.append((self.children_right[node_id], parent_depth + 1))
		    else:
		        is_leaves[node_id] = True

		print("The binary tree structure has %s nodes and has "
		      "the following tree structure:"
		      % self.num_nodes)
		for i in range(self.num_nodes):
		    if is_leaves[i]:
		        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
		    else:
		        print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
		              "node %s."
		              % (node_depth[i] * "\t",
		                 i,
		                 self.children_left[i],
		                 self.feature[i],
		                 self.threshold[i],
		                 self.children_right[i],
		                 ))
		return

	def test(self):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=0)
		
		leave_id = self.model.apply(X_test)
		data = {}
		print X_test
		for i in leave_id:
			data[leave_id[i]] = X_test[i]

		print data

	def produce_image(self):
		tree.export_graphviz(self.model, out_file='tree.dot', class_names=self.y)
		os.system("dot -Tpng tree.dot -o tree.png")
		os.system("open -a 'Adobe Photoshop CS6' tree.png")
		return

	def treeToJson(self, feature_names=None):
		decision_tree = self.model
		from warnings import warn

		js = ""

		def node_to_str(tree, node_id, criterion):
			if not isinstance(criterion, sklearn.tree.tree.six.string_types):
				criterion = "impurity"

			value = tree.value[node_id]
			if tree.n_outputs == 1:
				value = value[0, :]

			jsonValue = ', '.join([str(x) for x in value])

			if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
				return '"id": "%s", "criterion": "%s", "impurity": "%s", "samples": "%s", "value": [%s]' \
							% (node_id, 
							criterion,
							tree.impurity[node_id],
							tree.n_node_samples[node_id],
							jsonValue)
			else:
				if feature_names is not None:
					feature = feature_names[tree.feature[node_id]]
				else:
					feature = tree.feature[node_id]

				print feature
				ruleType = "<="
				ruleValue = "%.4f" % tree.threshold[node_id]
				# if "=" in feature:
				# 	ruleType = "="
				# 	ruleValue = "false"
				# else:
				# 	ruleType = "<="
				# 	ruleValue = "%.4f" % tree.threshold[node_id]

				return '"id": "%s", "rule": "%s %s %s", "%s": "%s", "samples": "%s"' \
						% (node_id, 
						feature,
						ruleType,
						ruleValue,
						criterion,
						tree.impurity[node_id],
						tree.n_node_samples[node_id])

		def recurse(tree, node_id, criterion, parent=None, depth=0):
			tabs = "  " * depth
			js = ""

			left_child = tree.children_left[node_id]
			right_child = tree.children_right[node_id]

			js = js + "\n" + \
				tabs + "{\n" + \
				tabs + "  " + node_to_str(tree, node_id, criterion)

			if left_child != sklearn.tree._tree.TREE_LEAF:
				js = js + ",\n" + \
					tabs + '  "left": ' + \
					recurse(tree, \
							left_child, \
							criterion=criterion, \
							parent=node_id, \
							depth=depth + 1) + ",\n" + \
					tabs + '  "right": ' + \
					recurse(tree, \
							right_child, \
							criterion=criterion, \
							parent=node_id,
							depth=depth + 1)

			js = js + tabs + "\n" + \
				tabs + "}"

			return js

		if isinstance(decision_tree, sklearn.tree.tree.Tree):
			js = js + recurse(decision_tree, 0, criterion="impurity")
		else:
			js = js + recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

		print js

	def rules(self, node_index=0):
	    """Structure of rules in a fit decision tree classifier

	    Parameters
	    ----------
	    clf : DecisionTreeClassifier
	        A tree that has already been fit.

	    features, labels : lists of str
	        The names of the features and labels, respectively.

	    """
	    clf = self.model
	    features = self.target
	    labels = self.y
	    node = {}
	    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
	        count_labels = zip(clf.tree_.value[node_index, 0], labels)
	        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
	                                  for count, label in count_labels))
	    else:
	        feature = features[clf.tree_.feature[node_index]]
	        threshold = clf.tree_.threshold[node_index]
	        node['name'] = '{} > {}'.format(feature, threshold)
	        left_index = clf.tree_.children_left[node_index]
	        right_index = clf.tree_.children_right[node_index]
	        node['children'] = [self.rules(right_index),
	                            self.rules(left_index)]
	    return node

	def get_question(self, iden):

		option = {
		# Frequent urination
			"Frequent urination":"Do you urinate frequently?",
		# "Do you urinate frequency?"
		# Hoarse voice
			"Hoarse voice":"Is your voice hoarse?",
		# Lacrimation
			"Lacrimation":"Do your eyes tear up excessively?",
		# Stomach bloating
			"Stomach bloating":"Is your stomach bloated?",
		# Too little hair
			"Too little hair":"Is there little hair in the region?",
		# Melena
			"Melena":"Is your stool black in color?",
		# Leg swelling
			"Leg swelling":"Is your leg(s) swollen?",
		# Burning chest pain
			"Burning chest pain":"Are you experiencing burning chest pain?",
		# Jaundice
			"Jaundice":"Is a portion of your skin yellow?",
		# Pain in testicles
			"Pain in testicles":"Are you experiencing pain in your testicles?",
		# Decreased appetite
			"Decreased appetite":"Do you have decreased appetite?",
		# Hemoptysis
			"Hemoptysis":"Are you coughing blood?",
		# Lip swelling
			"Lip swelling":"Is your lip swollen?",
		# Back pain
			"Back pain":"Do you have back pain?",
		# Elbow swelling
			"Elbow swelling":"Is there swelling in your elbow?",
		# Mouth symptoms
			"Mouth symptoms":"Is it in your mouth?",
		# Problems with movement
			"Problems with movement":"Are you experiencing problems with movement",
		# Weakness
			"Weakness":"Are you feeling weak?",
		# Itchy scalp
			"Itchy scalp":"Is your scalp itchy?",
		# Dizziness
			"Dizziness":"Do you feel dizzy?",
		# Vomiting blood
			"Vomiting blood":"Are you vomiting blood?",
		# Ear symptoms
			"Ear symptoms":"Are the symptoms in your ear?",
		# Palpitations
			"Palpitations":"Are you experiencing palpitations?",
		# Drug abuse
			"Drug abuse":"Do you use drugs?",
		# Mouth pain
			"Mouth pain":"Are you experiencing mouth pain?",
		# Feeling cold
			"Feeling cold":"Do you constantly feel cold?",
		# Infertility
			"Infertility":"Are you infertile?",
		# Pain of the anus
			"Pain of the anus":"Are you experiencing pain around your anus?",
		# Difficulty eating
			"Difficulty eating":"Do you find it difficult to eat?",
		# Abnormal involuntary movements
			"Abnormal involuntary movements":"Are you experiencing abnormal involuntary movement?",
		# Flu-like syndrome
			"Flu-like syndrome":"Do you feel 'flu-ish'?",
		# Loss of sensation
			"Loss of sensation":"Are you experiencing loss of sensation in the area?",
		# "Muscle cramps, contractures, or spasms"
			"Muscle cramps, contractures, or spasms":"Are you experiencing muscle cramps, muscle contractions or spasms?",
		# Frontal headache
			"Frontal headache":"Are you experiencing headaches at the front of your head?",
		# Suprapubic pain
			"Suprapubic pain":"Are you experiencing at the front of your pelvis?",
		# Depressive or psychotic symptoms
			"Depressive or psychotic symptoms":"Do you often feel severly depressed or distant from other people?",
		# Knee swelling
			"Knee swelling":"Is there swelling in your knee?",
		# Blood in stool
			"Blood in stool":"Is there blood in your stool?",
		# Back mass or lump
			"Back mass or lump":"Is there a lump or mass on your back?",
		# Fainting
			"Fainting":"Have you recently fainted?",
		# Mouth dryness
			"Mouth dryness":"Is your mouth dry?",
		# Mass on eyelid
			"Mass on eyelid":"Is there a lump on your eyelid?",
		# Acne or pimples
			"Acne or pimples":"Is there acne/pimples in the are?",
		# Arm weakness
			"Arm weakness":"Is there weakness in your arm?",
		# Ringing in ear
			"Ringing in ear":"Are your ears ringing?",
		# Hand or finger pain
			"Hand or finger pain":"Do you have pain in your hands or fingers?",
		# Foot or toe swelling
			"Foot or toe swelling":"Do you have swelling in your foot or toes?",
		# Decreased heart rate
			"Decreased heart rate":"Do you have a decreased heart rate?",
		# Incontinence of stool
			"Incontinence of stool":"Have you lost control of your stool?",
		# Sinus problems
			"Sinus problems":"Do you have sinus problems?",
		# Pain or soreness of breast
			"Pain or soreness of breast":"Are your breasts sore?",
		# Low urine output
			"Low urine output":"Do you have low urine output?",
		# Mouth ulcer
			"Mouth ulcer":"Do you have a sore (or ulcer) inside your mouth?",
		# Lower abdominal pain
			"Lower abdominal pain":"Do you have lower abdominal pain?",
		# Skin growth
			"Skin growth":"Are there callus', corns or skin tags in the area?",
		# Unusual color or odor to urine
			"Unusual color or odor to urine":"Is there an Unusual colour or odor to your urine?",
		# Obsessions and compulsions
			"Obsessions and compulsions":"Do you have obsessiosn and compulsions?",
		# 57Neck mass
			"Neck mass":"Is there a mass in your neck?",
		# 58Low back symptoms
			"Low back symptoms":"Are the symptoms in your lower back?",
		# 59Eye redness
			"Eye redness":"Are your eyes red?",
		# 60Visual disturbance
			"Visual disturbance":"Are you experiencing visual disturbances?",
		# 61Vaginal bleeding after menopause
			"Vaginal bleeding after menopause":"Do you have vaginal bleeding after menopause?",
		# 62Headache
			"Headache":"Are you experiencing headaches?",
		# 63Diminished hearing
			"Diminished hearing":"Has your hearing diminished?",
		# 64Long menstrual periods
			"Long menstrual periods":"Do you have long menstrual periods?",
		# 65Irregular appearing scalp
			"Irregular appearing scalp":"Do you have a dry scalp and/or a rash on your scalp?",
		# 66Hand or finger lump or mass
			"Hand or finger lump or mass":"Is there a lump present on your hands or fingers?",
		# 67Leg cramps or spasms
			"Leg cramps or spasms":"Are you experiencing leg cramps or spasms?",
		# 68Difficulty in swallowing
			"Difficulty in swallowing":"Are you having difficulty swallowing?",
		# 69Hysterical behavior
			"Hysterical behavior":"Are you behaving hysterically?",
		# 70Coughing up sputum
			"Coughing up sputum":"Are you coughing up mucus?",
		# 71Neck swelling
			"Neck swelling":"Is your neck swollen?",
		# 72Sleepiness
			"Sleepiness":"Do you feel drowsy/sleepy?",
		# 73Vaginal bleeding
			"Vaginal bleeding":"Do you have vaginal bleeding?",
		# 74Irregular belly button
			"Irregular belly button":"Is your belly button protruding irregularly?",
		# 75Seizures
			"Seizures":"Are you experiencing seizures?",
		# 76Irregular appearing nails
			"Irregular appearing nails":"Do you have irregularly appearing nails?",
		# 77Blood clots during menstrual periods
			"Blood clots during menstrual periods":"Do you have bloog clots during mentrual periods?",
		# 78Menopausal symptoms
			"Menopausal symptoms":"Are you undergoing menopause?",
		# 79Foot and toe symptoms
			"Foot and toe symptoms":"Are the symptoms present in your foot or toes?",
		# 80Ache all over
			"Ache all over":"Do you ache all over?",
		# 81Symptoms of eye
			"Symptoms of eye":"Are the symptoms in your eye(s)?",
		# 82Swelling of scrotum
			"Swelling of scrotum":"Is your scrotum swollen?",
		# 83Groin pain
			"Groin pain":"Are you experiencing groin pain?",
		# 84Spotting or bleeding during pregnancy
			"Spotting or bleeding during pregnancy":"Are you experiencing spotting or bleeding while pregnant?",
		# 85Elbow symptoms
			"Elbow symptoms":"Are the symptoms in or around your elbow?",
		# 86Nasal congestion
			"Nasal congestion":"Do you have nasal congestion?",
		# 87Abnormal breathing sounds
			"Abnormal breathing sounds":"Do you have abnormal breating sounds?",
		# 88Skin on leg or foot looks infected
			"Skin on leg or foot looks infected":"Does the skin on your leg or foot look infected?",
		# 89Bedwetting
			"Bedwetting":"Are you wetting the bed?",
		# 90Gum pain
			"Gum pain":"Are you experiencing pain or soreness in your gums?",
		# 91Fatigue
			"Fatigue":"Are you tired, fatigued or feel as if you have no energy?",
		# 92Excessive appetite
			"Excessive appetite":"Are you constantly hungry?",
		# 93Excessive urination at night
			"Excessive urination at night":"Do you excessively urinate at night?",
		# 94Joint pain
			"Joint pain":"Are you experiencing joint pain?",
		# 95Hostile behavior
			"Hostile behavior":"Do you behave violently or agressively?",
		# 96Wrist pain
			"Wrist pain":"Do you have wrist pain?",
		# 97Neurological symptoms
			"Neurological symptoms":"Are you experiencing neurological symptoms?",
		# 98Heartburn
			"Heartburn":"Do you have heartburn?",
		# 99Stiffness all over
			"Stiffness all over":"Do you feel stiff all over?",
		# 100 Pelvic pressure
			"Pelvic pressure":"Do you feel pressure in your pelvic region?",
		# 101: Shoulder symptoms
			"Shoulder symptoms":"Are the symptoms in your shoulder?",
		# 102: Painful menstruation
			"Painful menstruation":"Do you experience painful mentruations?",
		# 103: Rectal bleeding
			"Rectal bleeding":"Are you bleeding from your rectum?",
		# 104: Throat swelling
			"Throat swelling":"Is your throat swollen?",
		# 105: Thirst
			"Thirst":"Are you constantly thirsty?",
		# 106: Musculoskeletal deformities
			"Musculoskeletal deformities":"Do you have a crooked back?",
		# 107: Swollen or red tonsils
			"Swollen or red tonsils":"Do you have swollen or red tonsils?",
		# 108: Leg symptoms
			"Leg symptoms":"Are you experiencing your symptoms in your leg?",
		# 109: Shoulder stiffness or tightness
			"Shoulder stiffness or tightness":"Is your shoulder stiff or tight?",
		# 110: Leg lump or mass
			"Leg lump or mass":"Do you have a lump or mass in your leg?",
		# 111: Hand or finger weakness
			"Hand or finger weakness":"Do your hands or fingers feel weak?",
		# 112: White discharge from eye
			"White discharge from eye":"Do you have white discharge from your eye?",
		# 113: Pain during pregnancy
			"Pain during pregnancy":"Are you experiencing pain while pregnant?",
		# 114: Back symptoms
			"Back symptoms":"Are the symptoms you are experiencing in your back?",
		# 115: Itchiness of eye
			"Itchiness of eye":"Are your eye(s) itchy?",
		# 116: Sneezing
			"Sneezing":"Are you constantly sneezing?",
		# 117: Double vision
			"Double vision":"Do you have double vision?",
		# 118: Redness in or around nose
			"Redness in or around nose":"Do you have redness in or around your nose?",
		# 119: Penis symptoms
			"Penis symptoms":"Are the symptoms you are experiencing around your penis?",
		# 120: Problems during pregnancy
			"Problems during pregnancy":"Are you experiencing problems while pregnant?",
		# 121: Eye burns or stings
			"Eye burns or stings":"Does your eye burn or sting?",
		# 122: Swollen lymph nodes
			"Swollen lymph nodes":"Do you have swollen lymph nodes?",
		# 123: Ankle symptoms
			"Ankle symptoms":"Are you experiencing ankle symptoms?",
		# 124: Vaginal itching
			"Vaginal itching":"Are you experiencing vaginal itchiness?",
		# 125: Eye discharge
			"Eye discharge":"Is there discharge coming out of your eye(s)?",
		# 126: Sweating
			"Sweating":"Are you constantly sweating and/or do you have experience cold sweats?",
		# 127: Bleeding or discharge from nipple
			"Bleeding or discharge from nipple":"Are you bleeding and/or is there discharge from your nipple?",
		# 128: Side pain
			"Side pain":"Are you experiencing side pain?",
		# 129: Arm lump or mass
			"Arm lump or mass":"Do you have a lump or mass in your arm?",
		# 130: Skin looks infected
			"Skin looks infected":"Does your skin look infected?",
		# 131: Involuntary urination
			"Involuntary urination":"Are you experiencing involuntary urination?",
		# 132: Hand or finger swelling
			"Hand or finger swelling":"Is your hand or finger swollen?",
		# 133: Low self-esteem
			"Low self-esteem":"Do you have low self-esteem?",
		# 134: Spots or clouds in vision
			"Spots or clouds in vision":"Do you have spots or clouds in your vision?",
		# 135: Cramps and spasms
			"Cramps and spasms":"Are you experiencing cramps and spasms?",
		# 136: Arm stiffness or tightness
			"Arm stiffness or tightness":"Is your arm stiff or tight?",
		# 137: Lump or mass of breast
			"Lump or mass of breast":"Is there a lump or mass in your breast?",
		# 138: Pelvic pain
			"Pelvic pain":"Are you experiencing pelvic pain?",
		# 139: Itching of skin
			"Itching of skin":"Is you skin itchy?",
		# 140: Jaw swelling
			"Jaw swelling":"Is your jaw swollen?",
		# 141: Sharp abdominal pain
			"Sharp abdominal pain":"Are you experiencing sharp abdominal pain?",
		# 142: Symptoms of the female reproductive system
			"Symptoms of the female reproductive system":"Are your symptoms in your genital region?",
		# 143: Tongue lesions
			"Tongue lesions":"Do you have tongue lesions?",
		# 144: Abnormal appearing skin
			"Abnormal appearing skin":"Do you have abnormally appearing skin?",
		# 145: Symptoms of bladder
			"Symptoms of bladder":"Are the symptoms you are experiencing in your bladder?",
		# 146: Vaginal pain
			"Vaginal pain":"Do you have vaginal pain?",
		# 147: Arm symptoms
			"Arm symptoms":"Do you have arm symptoms?",
		# 148: Lump in throat
			"Lump in throat":"Do you have a lump in your throat?",
		# 149: Neck pain
			"Neck pain":"Are you experiencing neck pain?",
		# 150: Blood in urine
			"Blood in urine":"Do you have blood in your urine?",
		# 151: Leg pain
			"Leg pain":"Are you experiencing leg pain?",
		# 152: "Skin dryness, peeling, scaliness, or roughness"
			"Skin dryness, peeling, scaliness, or roughness":"Do you have skin dryness, peeling, scaliness or roughness?",
		# 153: Neck stiffness or tightness
			"Neck stiffness or tightness":"Do you have neck stiffnes or tightness?",
		# 154: Breathing fast
			"Breathing fast":"Are you rapidly breathing?",
		# 155: Sharp chest pain
			"Sharp chest pain":"Are you experiencing chest pain?",
		# 156: Uterine contractions
			"Uterine contractions":"Are you experiencing uterine contractions?",
		# 157: Warts
			"Warts":"Do you have warts?",
		# 158: Delusions or hallucinations
			"Delusions or hallucinations":"Are you experiencing delusions or hallucinations?",
		# 159: Ankle swelling
			"Ankle swelling":"Are your ankles swollen?",
		# 160: Disturbance of memory
			"Disturbance of memory":"Are you experiencing memory disturbances?",
		# 161: Lymphedema
			"Lymphedema":"Do you have swollen lymph nodes or vericose veins?",
		# 162: Eyelid swelling
			"Eyelid swelling":"Is your eyelid swollen?",
		# 163: Blindness
			"Blindness":"Are you experiencing blindness?",
		# 164: Diarrhea
			"Diarrhea":"Do you have Diarrhea?",
		# 165: Groin mass
			"Groin mass":"Is there a mass in your groin?",
		# 166: Vaginal discharge
			"Vaginal discharge":"Do you have vaginal discharge?",
		# 167: Swollen eye
			"Swollen eye":"Is your eye swollen?",
		# 168: Ear pain
			"Ear pain":"Are you experiencign ear pain?",
		# 169: Sore throat
			"Sore throat":"Do you have a sore throat?",
		# 170: Peripheral edema
			"Peripheral edema":"Are both of your ankles or both of your legs swelling?",
		# 171: Antisocial behavior
			"Antisocial behavior":"Do you have antisocial behaviour?",
		# 172: Neck symptoms
			"Neck symptoms":"Are the symptoms you are experiencing in your neck?",
		# 173: Low back pain
			"Low back pain":"Are you experiencing lower back pain?",
		# 174: Skin swelling
			"Skin swelling":"Are you experiencing skin swelling?",
		# 175: Symptoms of the face
			"Symptoms of the face":"Are your symptoms in your face?",
		# 176: Restlessness
			"Restlessness":"Are you constantly restless?",
		# 177: Pain during intercourse
			"Pain during intercourse":"Do you experience pain during intercourse?",
		# 178: Penis redness
			"Penis redness":"Is your penis appearing red?",
		# 179: Fears and phobias
			"Fears and phobias":"Do you have intense fears and phobias?",
		# 180: Changes in stool appearance
			"Changes in stool appearance":"Has there recently been changes in your stool appearance or colour?",
		# 181: Constipation
			"Constipation":"Are you experiencing constipation?",
		# 182: Throat feels tight
			"Throat feels tight":"Does your throat feel tight?",
		# 183: Skin pain
			"Skin pain":"Are you experiencing skin pain?",
		# 184: Difficulty speaking
			"Difficulty speaking":"Are you having difficulty speaking?",
		# 185: Nausea
			"Nausea":"Do you feel nauseous?",
		# 186: Arm swelling
			"Arm swelling":"Is your arm swollen?",
		# 187: Lack of growth
			"Lack of growth":"Are you experiencing lack of growth?",
		# 188: Burning abdominal pain
			"Burning abdominal pain":"Are you experiencing burning abdominal pain?",
		# 189: Leg weakness
			"Leg weakness":"Does your leg feel weak?",
		# 190: Leg stiffness or tightness
			"Leg stiffness or tightness":"Does your leg feel stiff or tight?",
		# 191: Weight loss
			"Weight loss":"Have you recently lost weight?",
		# 192: Behavioral disturbances
			"Behavioral disturbances":"Do you feel agitated, a lack of control or have a similar behavior problem?",
		# 193: Sinus congestion
			"Sinus congestion":"Are you experiencing sinus congestion?",
		# 194: Wheezing
			"Wheezing":"Are you wheezing?",
		# 195: Difficulty breathing
			"Difficulty breathing":"Do you have difficulty breathing?",
		# 196: Sleep disturbance
			"Sleep disturbance":"Do you have frequent nightmares/ night terrors?",
		# 197: Hand or finger stiffness or tightness
			"Hand or finger stiffness or tightness":"Do your hands/ fingers feel stiff or tight?",
		# 198: Wrist swelling
			"Wrist swelling":"Is your wrist swollen?",
		# 199: Excessive anger
			"Excessive anger":"Are you excessively angry/agresive?",
		# 200: Facial pain
			"Facial pain":"Are you experiencing facial pain?",
		# 201: Pulling at ears
			"Pulling at ears":"Do you feel as if your ears are being pulled?",
		# 202: Vaginal symptoms
			"Vaginal symptoms":"Are the symptosm you are experiencing in your vaginal region?",
		# 203: Congestion in chest
			"Congestion in chest":"Are you experiencing chest congestion?",
		# 204: Problems with shape or size of breasts
			"Problems with shape or size of breasts":"Are there problems with the size or shape of your breasts?",
		# 205: Insomnia
			"Insomnia":"Do you have difficulty sleeping?",
		# 206: Knee weakness
			"Knee weakness":"Does your knee feel weak?",
		# 207: Back cramps or spasms
			"Back cramps or spasms":"Are you experiencign back cramps or spasms?",
		# 208: Bleeding from ear
			"Bleeding from ear":"Are you bleeding from your ear?",
		# 209: Diaper rash
			"Diaper rash":"Do you have a diaper rash?",
		# 210: Changes in bowel function
			"Changes in bowel function":"Have you experienced recent changes in your bowel function?",
		# 211: Skin lesion
			"Skin lesion":"Is ther a lesion/sore on your skin?",

		# 212: Painful sinuses
			"Painful sinuses":"Are you experiencing painful sinuses?",
		# 213: Skin moles
			"Skin moles":"Are there moles on your skin?",
		# 214: Itchy ear(s)
			"Itchy ear(s)":"Are your ears itchy?",
		# 215: Swollen tongue
			"Swollen tongue":"Is your tongue swollen?",
		# 216: Eyelid lesion or rash
			"Eyelid lesion or rash":"Is there a lesion/sore or rash on your eyelid?",
		# 217: Penile discharge
			"Penile discharge":"Do you have discharge from your penis?",
		# 218: Feeling hot
			"Feeling hot":"Do you feel hot?",
		# 219: Increased heart rate
			"Increased heart rate":"Is your heart racing?",
		# 220: Chest pain
			"Chest pain":"Do you have chest pain?",
		# 221: Hip symptoms
			"Hip symptoms":"Are your symptoms in your hip?",
		# 222: Infant spitting up
			"Infant spitting up":"Is your infant spitting up?",
		# 223: Irregular heartbeat
			"Irregular heartbeat":"Do you have an irregular heartbeat?",
		# 224: Hip pain
			"Hip pain":"Are you experiencing hip pain?",
		# 225: Diminished vision
			"Diminished vision":"Do you have blurred vison, have trouble reading or trouble focusing?",
		# 226: Symptoms of the breast
			"Symptoms of the breast":"Are your symptoms in your breast?",
		# 227: Lower body pain
			"Lower body pain":"Are you experiencing lower body pain?",
		# 228: Skin on arm or hand looks infected
			"Skin on arm or hand looks infected":"Does the skin on your arm or hand look infected?",
		# 229: Elbow pain
			"Elbow pain":"Do you ave pain in your elbow?",
		# 230: Impotence
			"Impotence":"Are you unable to get an erection?",
		# 231: Slurring words
			"Slurring words":"Do you slur your words?",
		# 232: Allergic reaction
			"Allergic reaction":"Do you have hives and/or trouble breathing?",
		# 233: Recent pregnancy
			"Recent pregnancy":"Were you recently pregnant?",
		# 234: Anxiety and nervousness
			"Anxiety and nervousness":"Are you anxious or nervous?",
		# 235: Intermenstrual bleeding
			"Intermenstrual bleeding":"Are you bleeding in between menstrual periods?",
		# 236: Hurts to breath
			"Hurts to breath":"Does it hurt to breath?",
		# 237: Foot or toe pain
			"Foot or toe pain":"Do you have foot or toe pain?",
		# 238: Temper problems
			"Temper problems":"Do you have a temper problem/ are you angry often?",
		# 239: Painful urination
			"Painful urination":"Is it painful to urinate?",
		# 240: Fever
			"Fever":"Do you feel feverish/ are you warm to the touch?",
		# 241: Absence of menstruation
			"Absence of menstruation":"Have you not had a menstrual cycle in a while?",
		# 242: Knee symptoms
			"Knee symptoms":"Are you experiencing knee symptoms?",
		# 243: Irritable infant
			"Irritable infant":"Do you have an irritable infant?",
		# 244: Nosebleed
			"Nosebleed":"Do you have a nosebleed?",
		# 245: Wrist symptoms
			"Wrist symptoms":"Do you have wrist symptoms?",
		# 246: Unpredictable menstruation
			"Unpredictable menstruation":"Is your menstrual cycle unpredictable?",
		# 247: Shoulder pain
			"Shoulder pain":"Do you have shoulder pain?",
		# 248: Smoking problems
			"Smoking problems":"Do you smoke a lot?",
		# 249: Paresthesia
			"Paresthesia":"Do you feel burning or prickling on your skin?",
		# 250: Toothache
			"Toothache":"Do you have a toothache?",
		# 251: Arm pain
			"Arm pain":"Do you have arm pain?",
		# 252: Pelvic symptoms
			"Pelvic symptoms":"Are your symptoms in or around your pelvis?",
		# 253: Foreign body sensation in eye
			"Foreign body sensation in eye":"Do you feel like there is something in your eye?",
		# 254: Chest tightness
			"Chest tightness":"Are you experiencing tightness in your chest?",
		# 255: Ankle pain
			"Ankle pain":"Do you have ankle pain?",
		# 256: Hand and finger symptoms
			"Hand and finger symptoms":"Are your symptoms in your hand/finger?",
		# 257: Abnormal movement of eyelid
			"Abnormal movement of eyelid":"Are you excessively blinking, is your eyelid drooping or are you squinting excessively?",
		# 258: Feeling ill
			"Feeling ill":"Do you generall feel ill?",
		# 259: Heavy menstrual flow
			"Heavy menstrual flow":"Do you have a heavy mentrual flow?",
		# 260: Bones are painful
			"Bones are painful":"Are your bones painful?",
		# 261: Apnea
			"Apnea":"Do you snore loudly or stop breathing while you sleep?",
		# 262: Throat irritation
			"Throat irritation":"Does your through feel itchy/scratchy?",
		# 263: Symptoms of prostate
			"Symptoms of prostate":"Are your symptoms in or around your prostate?",
		# 264: Mass in scrotum
			"Mass in scrotum":"Do you have a mass in your scrotum?",
		# 265: Symptoms of the skin
			"Symptoms of the skin":"Are your symptoms on your skin?",
		# 266: Focal weakness
			"Focal weakness":"Do you feel weak on one side of your body?",
		# 267: Abnormal growth or development
			"Abnormal growth or development":"Are you experiecning abnormal growth or development?",
		# 268: Disturbance of smell or taste
			"Disturbance of smell or taste":"Do you have problems with your sense of smell or taste?",
		# 269: Nose symptoms
			"Nose symptoms":"Are your symptoms in your nose?",
		# 270: Knee stiffness or tightness
			"Knee stiffness or tightness":"Are you experiencing knee stiffness or tightness?",
		# 271: Plugged feeling in ear
			"Plugged feeling in ear":"Does your ear(s) feel plugged?",
		# 272: Symptoms of the kidneys
			"Symptoms of the kidneys":"Are your symptoms in your kidney/lower back?",
		# 273: Symptoms of the anus
			"Symptoms of the anus":"Are your symptoms in or around your anus?",
		# 274: Cough
			"Cough":"Do you have a cough?",
		# 275: Regurgitation
			"Regurgitation":"Are you regurgitating?",
		# 276: Coryza
			"Coryza":"Do you have a stuffy nose?",
		# 277: Pain in eye
			"Pain in eye":"Do you have pain in your eye?",
		# 278: Vomiting
			"Vomiting":"Are you vomiting?",
		# 279: Fluid retention
			"Fluid retention":"Are you retaining a lot of fluid?",
		# 280: Hot flashes
			"Hot flashes":"Are you experiencing hot flashes?",
		# 281: Upper abdominal painful
			"Upper abdominal painful":"Do you have upper abdominal pain?",
		# 282: Skin rash
			"Skin rash":"Do you have a skin rash?",
		# 283: Rib pain
			"Rib pain":"Do you have pain in your ribs?",
		# 284: Swollen abdomen
			"Swollen abdomen":"Is your abdomen swollen?",
		# 285: Infant feeding problem
			"Infant feeding problem":"Are you having trouble feeding your infant?",
		# 286: Knee pain
			"Knee pain":"Do you have knee pain?",
		# 287: Redness in ear
			"Redness in ear":"Is there redness in your ear?",
		# 288: Flatulence
			"Flatulence":"Are you overly flatulent?",
		# 289: Chills
			"Chills":"Do you have chills?",
		# 290: Depression
			"Depression":"Are you depressed/ are you sad all of the time?",
		# 291: Weight gain
			"Weight gain":"Have you recently undergone weight gain?",
		# 292: Muscle pain
			"Muscle pain":"Do you have pain in your muscles?",
		# 293: Fluid in ear
			"Fluid in ear":"Is there fluid in your ear. does your ear feel full?",
		# 294: Penis pain
			"Penis pain":"Does your penis hurt?",
		# 295: Abusing alcohol
			"Abusing alcohol":"Do you drink alcohol excessively?",
		# 296: Retention of urine
			"Retention of urine":"Are you unable to urinate or do you have trouble urinating?",
		# 297: Drainage in throat
			"Drainage in throat":"Is your nose draining mucus into your throat?",
		# 298: Skin irritation
			"Skin irritation":"Is your skin irritated/ do you have a rash?",
		# 299: Shortness of breath
			"Shortness of breath":"Are you experiencing shortness of breath?",
			"Female": "Are you a female?",
			"Male" : "Are you a male?",
			"Upper Body": "Are the symptoms in your upper body (above your pelvis)?",
			"Lower Body": "Are the symptoms in your lower body (pelvis or below)?",
			"Skin": "Are you experiencing symptoms on/in your skin?",
			"Other": "Are you're symtoms not specifically at any location on your body?",
			"Face and Head": "Are your symptoms in or around your head and/or face?",
			"Neck":"Are your symptoms in or around your neck?",
			"Throat":"Are your symptoms in or around your throat?",
			"Back Neck": "Are the symptoms you are experiencing at the back of your neck?",
			"Shoulders": "Are the symptoms you are experiencing in your shoulders?",
			"Arms":"Are you experiencing symptoms in your arms?",
			"Hands":"Are you experiencing symptoms in your hands?",
			"Back":"Are the symtoms you are experiecing in or around your back?",
			"Upper Back":"Are the symptoms in your upper back?",
			"Lower Back":"Are the symptoms in your lower back?",
			"Brain":"Do you feel as if the symptoms you are experiencing are inside your head/ in your brain?",
			"Skull":"Do you feel symptoms in your skull or scalp?",
			"Abdomen":"Are your symptoms in your abdomen?",
			"Upper Ab":"Are your symptoms in your upper abdomen?",
			"Lower Ab":"Are your symptoms in your lower abdomen?",
			"Stomach":"Are your symptoms in your stomach?",
			"Chest":"Are your symptoms in or around your chest?",
			"Pelvis":"Are your symptoms in or around your pelvic region",
			"Genitals":"Are your symptoms in or aroudn your genitals?",
			"Anus":"Are your symptoms in or around your anus?",
			"Legs":"Are your symptoms in or around your legs?",
			"Upper Leg":"Are your symptoms in your upper leg?",
			"Lower Leg":"Are your symptoms in your lower leg",
			"Feet":"Are your symptoms in your feet?",
			"Joints":"Are your symptoms in your joints?",
			"Lymph nodes":"Are your symptoms in your lymph nodes?",
			"Bones":"Are your symptoms in your bones?",
			"Stool":"Are your symptoms in your stool?",
			"Muscle":"Are your symptoms relating to your muscle?",
			"Sinus":"Are your symptoms relating to your sinuses?",
			"Phycho/feeling": "Are your symptoms at no specific location in your body or are your symptoms mental?",
			"Infant":"Are your symptoms relating to a pregnancy or an infant?",
			"Nose":"Are your symptoms in your nose?",
			"Vommit":"Are your symptoms related to vommiting?",
			"Flatulence":"Are your symptoms related to flatulence?",
			"Eyes":"Are your symptoms in your eyes?",
			"Mouth":"Are your symptoms in your mouth?",
			"Ears":"Are your symptoms in your ears"
		}
		return option[iden]



