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
		# self.clean_target(target)
		X = X.astype(int)
		X = np.delete(X, 0, 0)
		y = np.delete(y, 0)
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
		model = DecisionTreeClassifier(criterion='entropy', presort=True)
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
		tree.export_graphviz(self.model, out_file='tree.dot', class_names=self.target)
		os.system("dot -Tpng tree.dot -o tree.png")
		os.system("open tree.png")
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

	def get_question(self, iden):

		option = {
		# Frequent urination
			0:"Do you urinate frequently?",
		# "Do you urinate frequency?"
		# Hoarse voice
			1:"Is your voice hoarse?",
		# Lacrimation
			2:"Do your eyes tear up excessively?",
		# Stomach bloating
			3:"Is your stomach bloated?",
		# Too little hair
			4:"Is there little hair in the region?",
		# Melena
			5:"Is your stool black in color?",
		# Leg swelling
			6:"Is your leg(s) swollen?",
		# Burning chest pain
			7:"Are you experiencing burning chest pain?",
		# Jaundice
			8:"Is a portion of your skin yellow?",
		# Pain in testicles
			9:"Are you experiencing pain in your testicles?",
		# Decreased appetite
			10:"Do you have decreased appetite?",
		# Hemoptysis
			11:"Are you coughing blood?",
		# Lip swelling
			12:"Is your lip swollen?",
		# Back pain
			13:"Do you have back pain?",
		# Elbow swelling
			14:"Is there swelling in your elbow?",
		# Mouth symptoms
			15:"Is it in your mouth?",
		# Problems with movement
			16:"Are you experiencing problems with movement",
		# Weakness
			17:"Are you feeling weak?",
		# Itchy scalp
			18:"Is your scalp itchy?",
		# Dizziness
			19:"Do you feel dizzy?",
		# Vomiting blood
			20:"Are you vomiting blood?",
		# Ear symptoms
			21:"Are the symptoms in your ear?",
		# Palpitations
			22:"Are you experiencing palpitations?",
		# Drug abuse
			23:"Do you use drugs?",
		# Mouth pain
			24:"Are you experiencing mouth pain?",
		# Feeling cold
			25:"Do you constantly feel cold?",
		# Infertility
			26:"Are you infertile?",
		# Pain of the anus
			27:"Are you experiencing pain around your anus?",
		# Difficulty eating
			28:"Do you find it difficult to eat?",
		# Abnormal involuntary movements
			29:"Are you experiencing abnormal involuntary movement?",
		# Flu-like syndrome
			30:"Do you feel 'flu-ish'?",
		# Loss of sensation
			31:"Are you experiencing loss of sensation in the area?",
		# "Muscle cramps, contractures, or spasms"
			32:"Are you experiencing muscle cramps, muscle contractions or spasms?",
		# Frontal headache
			33:"Are you experiencing headaches at the front of your head?",
		# Suprapubic pain
			34:"Are you experiencing at the front of your pelvis?",
		# Depressive or psychotic symptoms
			35:"Do you often feel severly depressed or distant from other people?",
		# Knee swelling
			36:"Is there swelling in your knee?",
		# Blood in stool
			37:"Is there blood in your stool?",
		# Back mass or lump
			38:"Is there a lump or mass on your back?",
		# Fainting
			39:"Have you recently fainted?",
		# Mouth dryness
			40:"Is your mouth dry?",
		# Mass on eyelid
			41:"Is there a lump on your eyelid?",
		# Acne or pimples
			42:"Is there acne/pimples in the are?",
		# Arm weakness
			43:"Is there weakness in your arm?",
		# Ringing in ear
			44:"Are your ears ringing?",
		# Hand or finger pain
			45:"Do you have pain in your hands or fingers?",
		# Foot or toe swelling
			46:"Do you have swelling in your foot or toes?",
		# Decreased heart rate
			47:"Do you have a decreased heart rate?",
		# Incontinence of stool
			48:"Have you lost control of your stool?",
		# Sinus problems
		# Pain or soreness of breast
		# Low urine output
		# Mouth ulcer
		# Lower abdominal pain
		# Skin growth
		# Unusual color or odor to urine
		# Obsessions and compulsions
		# Neck mass
		# Low back symptoms
		# Eye redness
		# Visual disturbance
		# Vaginal bleeding after menopause
		# Headache
		# Diminished hearing
		# Long menstrual periods
		# Irregular appearing scalp
		# Hand or finger lump or mass
		# Leg cramps or spasms
		# Difficulty in swallowing
		# Hysterical behavior
		# Coughing up sputum
		# Neck swelling
		# Sleepiness
		# Vaginal bleeding
		# Irregular belly button
		# Seizures
		# Irregular appearing nails
		# Blood clots during menstrual periods
		# Menopausal symptoms
		# Foot and toe symptoms
		# Ache all over
		# Symptoms of eye
		# Swelling of scrotum
		# Groin pain
		# Spotting or bleeding during pregnancy
		# Elbow symptoms
		# Nasal congestion
		# Abnormal breathing sounds
		# Skin on leg or foot looks infected
		# Bedwetting
		# Gum pain
		# Fatigue
		# Excessive appetite
		# Excessive urination at night
		# Joint pain
		# Hostile behavior
		# Wrist pain
		# Neurological symptoms
		# Heartburn
		# Stiffness all over
		# Pelvic pressure
		# Shoulder symptoms
		# Painful menstruation
		# Rectal bleeding
		# Throat swelling
		# Thirst
		# Musculoskeletal deformities
		# Swollen or red tonsils
		# Leg symptoms
		# Shoulder stiffness or tightness
		# Leg lump or mass
		# Hand or finger weakness
		# White discharge from eye
		# Pain during pregnancy
		# Back symptoms
		# Itchiness of eye
		# Sneezing
		# Double vision
		# Redness in or around nose
		# Penis symptoms
		# Problems during pregnancy
		# Eye burns or stings
		# Swollen lymph nodes
		# Ankle symptoms
		# Vaginal itching
		# Eye discharge
		# Sweating
		# Bleeding or discharge from nipple
		# Side pain
		# Arm lump or mass
		# Skin looks infected
		# Involuntary urination
		# Hand or finger swelling
		# Low self-esteem
		# Spots or clouds in vision
		# Cramps and spasms
		# Arm stiffness or tightness
		# Lump or mass of breast
		# Pelvic pain
		# Itching of skin
		# Jaw swelling
		# Sharp abdominal pain
		# Symptoms of the female reproductive system
		# Tongue lesions
		# Abnormal appearing skin
		# Symptoms of bladder
		# Vaginal pain
		# Arm symptoms
		# Lump in throat
		# Neck pain
		# Blood in urine
		# Leg pain
		# "Skin dryness, peeling, scaliness, or roughness"
		# Neck stiffness or tightness
		# Breathing fast
		# Sharp chest pain
		# Uterine contractions
		# Warts
		# Delusions or hallucinations
		# Ankle swelling
		# Disturbance of memory
		# Lymphedema
		# Eyelid swelling
		# Blindness
		# Diarrhea
		# Groin mass
		# Vaginal discharge
		# Swollen eye
		# Ear pain
		# Sore throat
		# Peripheral edema
		# Antisocial behavior
		# Neck symptoms
		# Low back pain
		# Skin swelling
		# Symptoms of the face
		# Restlessness
		# Pain during intercourse
		# Penis redness
		# Fears and phobias
		# Changes in stool appearance
		# Constipation
		# Throat feels tight
		# Skin pain
		# Difficulty speaking
		# Nausea
		# Arm swelling
		# Lack of growth
		# Burning abdominal pain
		# Leg weakness
		# Leg stiffness or tightness
		# Weight loss
		# Behavioral disturbances
		# Sinus congestion
		# Wheezing
		# Difficulty breathing
		# Sleep disturbance
		# Hand or finger stiffness or tightness
		# Wrist swelling
		# Excessive anger
		# Facial pain
		# Pulling at ears
		# Vaginal symptoms
		# Congestion in chest
		# Problems with shape or size of breast
		# Insomnia
		# Knee weakness
		# Back cramps or spasms
		# Bleeding from ear
		# Diaper rash
		# Changes in bowel function
		# Skin lesion
		# Painful sinuses
		# Skin moles
		# Itchy ear(s)
		# Swollen tongue
		# Eyelid lesion or rash
		# Penile discharge
		# Feeling hot
		# Increased heart rate
		# Chest pain
		# Hip symptoms
		# Infant spitting up
		# Irregular heartbeat
		# Hip pain
		# Diminished vision
		# Symptoms of the breast
		# Lower body pain
		# Skin on arm or hand looks infected
		# Elbow pain
		# Impotence
		# Slurring words
		# Allergic reaction
		# Recent pregnancy
		# Anxiety and nervousness
		# Intermenstrual bleeding
		# Hurts to breath
		# Foot or toe pain
		# Temper problems
		# Painful urination
		# Fever
		# Absence of menstruation
		# Knee symptoms
		# Irritable infant
		# Nosebleed
		# Wrist symptoms
		# Unpredictable menstruation
		# Shoulder pain
		# Smoking problems
		# Paresthesia
		# Toothache
		# Arm pain
		# Pelvic symptoms
		# Foreign body sensation in eye
		# Chest tightness
		# Ankle pain
		# Hand and finger symptoms
		# Abnormal movement of eyelid
		# Feeling ill
		# Heavy menstrual flow
		# Bones are painful
		# Apnea
		# Throat irritation
		# Symptoms of prostate
		# Mass in scrotum
		# Symptoms of the skin
		# Focal weakness
		# Abnormal growth or development
		# Disturbance of smell or taste
		# Nose symptoms
		# Knee stiffness or tightness
		# Plugged feeling in ear
		# Symptoms of the kidneys
		# Symptoms of the anus
		# Cough
		# Regurgitation
		# Coryza
		# Pain in eye
		# Vomiting
		# Fluid retention
		# Hot flashes
		# Upper abdominal pain
		# Skin rash
		# Rib pain
		# Swollen abdomen
		# Infant feeding problem
		# Knee pain
		# Redness in ear
		# Flatulence
		# Chills
		# Depression
		# Weight gain
		# Muscle pain
		# Fluid in ear
		# Penis pain
		# Abusing alcohol
		# Retention of urine
		# Drainage in throat
		# Skin irritation
		# Shortness of breath
		}
		return option[iden]



