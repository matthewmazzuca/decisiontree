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
			49:"Do you have sinus problems?",
		# Pain or soreness of breast
			50:"Are your breasts sore?",
		# Low urine output
			51:"Do you have low urine output?",
		# Mouth ulcer
			52:"Do you have a sore (or ulcer) inside your mouth?",
		# Lower abdominal pain
			53:"Do you have lower abdominal pain?",
		# Skin growth
			54:"Are there callus', corns or skin tags in the area?",
		# Unusual color or odor to urine
			55:"Is there an Unusual colour or odor to your urine?",
		# Obsessions and compulsions
			56:"Do you have obsessiosn and compulsions?",
		# 57Neck mass
			57:"Is there a mass in your neck?",
		# 58Low back symptoms
			58:"Are the symptoms in your lower back?",
		# 59Eye redness
			59:"Are your eyes red?",
		# 60Visual disturbance
			60:"Are you experiencing visual disturbances?",
		# 61Vaginal bleeding after menopause
			61:"Do you have vaginal bleeding after menopause?",
		# 62Headache
			62:"Are you experiencing headaches?",
		# 63Diminished hearing
			63:"Has your hearing diminished?",
		# 64Long menstrual periods
			64:"Do you have long menstrual periods?",
		# 65Irregular appearing scalp
			65:"Do you have a dry scalp and/or a rash on your scalp?",
		# 66Hand or finger lump or mass
			66:"Is there a lump present on your hands or fingers?",
		# 67Leg cramps or spasms
			67:"Are you experiencing leg cramps or spasms?",
		# 68Difficulty in swallowing
			68:"Are you having difficulty swallowing?",
		# 69Hysterical behavior
			69:"Are you behaving hysterically?",
		# 70Coughing up sputum
			70:"Are you coughing up mucus?",
		# 71Neck swelling
			71:"Is your neck swollen?",
		# 72Sleepiness
			72:"Do you feel drowsy/sleepy?",
		# 73Vaginal bleeding
			73:"Do you have vaginal bleeding?",
		# 74Irregular belly button
			74:"Is your belly button protruding irregularly?",
		# 75Seizures
			75:"Are you experiencing seizures?",
		# 76Irregular appearing nails
			76:"Do you have irregularly appearing nails?",
		# 77Blood clots during menstrual periods
			77:"Do you have bloog clots during mentrual periods?",
		# 78Menopausal symptoms
			78:"Are you undergoing menopause?",
		# 79Foot and toe symptoms
			79:"Are the symptoms present in your foot or toes?",
		# 80Ache all over
			80:"Do you ache all over?",
		# 81Symptoms of eye
			81:"Are the symptoms in your eye(s)?",
		# 82Swelling of scrotum
			82:"Is your scrotum swollen?",
		# 83Groin pain
			83:"Are you experiencing groin pain?",
		# 84Spotting or bleeding during pregnancy
			84:"Are you experiencing spotting or bleeding while pregnant?",
		# 85Elbow symptoms
			85:"Are the symptoms in or around your elbow?",
		# 86Nasal congestion
			86:"Do you have nasal congestion?",
		# 87Abnormal breathing sounds
			87:"Do you have abnormal breating sounds?",
		# 88Skin on leg or foot looks infected
			88:"Does the skin on your leg or foot look infected?",
		# 89Bedwetting
			89:"Are you wetting the bed?",
		# 90Gum pain
			90:"Are you experiencing pain or soreness in your gums?",
		# 91Fatigue
			91:"Are you tired, fatigued or feel as if you have no energy?",
		# 92Excessive appetite
			92:"Are you constantly hungry?",
		# 93Excessive urination at night
			93:"Do you excessively urinate at night?",
		# 94Joint pain
			94:"Are you experiencing joint pain?",
		# 95Hostile behavior
			95:"Do you behave violently or agressively?",
		# 96Wrist pain
			96:"Do you have wrist pain?",
		# 97Neurological symptoms
			97:"Are you experiencing neurological symptoms?",
		# 98Heartburn
			98:"Do you have heartburn?",
		# 99Stiffness all over
			99:"Do you feel stiff all over?",
		# 100 Pelvic pressure
			100:"Do you feel pressure in your pelvic region?",
		# 101: Shoulder symptoms
		# 102: Painful menstruation
		# 103: Rectal bleeding
		# 104: Throat swelling
		# 105: Thirst
		# 106: Musculoskeletal deformities
		# 107: Swollen or red tonsils
		# 108: Leg symptoms
		# 109: Shoulder stiffness or tightness
		# 110: Leg lump or mass
		# 111: Hand or finger weakness
		# 112: White discharge from eye
		# 113: Pain during pregnancy
		# 114: Back symptoms
		# 115: Itchiness of eye
		# 116: Sneezing
		# 117: Double vision
		# 118: Redness in or around nose
		# 119: Penis symptoms
		# 120: Problems during pregnancy
		# 121: Eye burns or stings
		# 122: Swollen lymph nodes
		# 123: Ankle symptoms
		# 124: Vaginal itching
		# 125: Eye discharge
		# 126: Sweating
		# 127: Bleeding or discharge from nipple
		# 128: Side pain
		# 129: Arm lump or mass
		# 130: Skin looks infected
		# 131: Involuntary urination
		# 132: Hand or finger swelling
		# 133: Low self-esteem
		# 134: Spots or clouds in vision
		# 135: Cramps and spasms
		# 136: Arm stiffness or tightness
		# 137: Lump or mass of breast
		# 138: Pelvic pain
		# 139: Itching of skin
		# 140: Jaw swelling
		# 141: Sharp abdominal pain
		# 142: Symptoms of the female reproductive system
		# 143: Tongue lesions
		# 144: Abnormal appearing skin
		# 145: Symptoms of bladder
		# 146: Vaginal pain
		# 147: Arm symptoms
		# 148: Lump in throat
		# 149: Neck pain
		# 150: Blood in urine
		# 151: Leg pain
		# 152: "Skin dryness, peeling, scaliness, or roughness"
		# 153: Neck stiffness or tightness
		# 154: Breathing fast
		# 155: Sharp chest pain
		# 156: Uterine contractions
		# 157: Warts
		# 158: Delusions or hallucinations
		# 159: Ankle swelling
		# 160: Disturbance of memory
		# 161: Lymphedema
		# 162: Eyelid swelling
		# 163: Blindness
		# 164: Diarrhea
		# 165: Groin mass
		# 166: Vaginal discharge
		# 167: Swollen eye
		# 168: Ear pain
		# 169: Sore throat
		# 170: Peripheral edema
		# 171: Antisocial behavior
		# 172: Neck symptoms
		# 173: Low back pain
		# 174: Skin swelling
		# 175: Symptoms of the face
		# 176: Restlessness
		# 177: Pain during intercourse
		# 178: Penis redness
		# 179: Fears and phobias
		# 180: Changes in stool appearance
		# 181: Constipation
		# 182: Throat feels tight
		# 183: Skin pain
		# 184: Difficulty speaking
		# 185: Nausea
		# 186: Arm swelling
		# 187: Lack of growth
		# 188: Burning abdominal pain
		# 189: Leg weakness
		# 190: Leg stiffness or tightness
		# 191: Weight loss
		# 192: Behavioral disturbances
		# 193: Sinus congestion
		# 194: Wheezing
		# 195: Difficulty breathing
		# 196: Sleep disturbance
		# 197: Hand or finger stiffness or tightness
		# 198: Wrist swelling
		# 199: Excessive anger
		# 200: Facial pain
		# 201: Pulling at ears
		# 202: Vaginal symptoms
		# 203: Congestion in chest
		# 204: Problems with shape or size of breast
		# 205: Insomnia
		# 206: Knee weakness
		# 207: Back cramps or spasms
		# 208: Bleeding from ear
		# 209: Diaper rash
		# 210: Changes in bowel function
		# 211: Skin lesion
		# 212: Painful sinuses
		# 213: Skin moles
		# 214: Itchy ear(s)
		# 215: Swollen tongue
		# 216: Eyelid lesion or rash
		# 217: Penile discharge
		# 218: Feeling hot
		# 219: Increased heart rate
		# 220: Chest pain
		# 221: Hip symptoms
		# 222: Infant spitting up
		# 223: Irregular heartbeat
		# 224: Hip pain
		# 225: Diminished vision
		# 226: Symptoms of the breast
		# 227: Lower body pain
		# 228: Skin on arm or hand looks infected
		# 229: Elbow pain
		# 230: Impotence
		# 231: Slurring words
		# 232: Allergic reaction
		# 233: Recent pregnancy
		# 234: Anxiety and nervousness
		# 235: Intermenstrual bleeding
		# 236: Hurts to breath
		# 237: Foot or toe pain
		# 238: Temper problems
		# 239: Painful urination
		# 240: Fever
		# 241: Absence of menstruation
		# 242: Knee symptoms
		# 243: Irritable infant
		# 244: Nosebleed
		# 245: Wrist symptoms
		# 246: Unpredictable menstruation
		# 247: Shoulder pain
		# 248: Smoking problems
		# 249: Paresthesia
		# 250: Toothache
		# 251: Arm pain
		# 252: Pelvic symptoms
		# 253: Foreign body sensation in eye
		# 254: Chest tightness
		# 255: Ankle pain
		# 256: Hand and finger symptoms
		# 257: Abnormal movement of eyelid
		# 258: Feeling ill
		# 259: Heavy menstrual flow
		# 260: Bones are painful
		# 261: Apnea
		# 262: Throat irritation
		# 263: Symptoms of prostate
		# 264: Mass in scrotum
		# 265: Symptoms of the skin
		# 266: Focal weakness
		# 267: Abnormal growth or development
		# 268: Disturbance of smell or taste
		# 269: Nose symptoms
		# 270: Knee stiffness or tightness
		# 271: Plugged feeling in ear
		# 272: Symptoms of the kidneys
		# 273: Symptoms of the anus
		# 274: Cough
		# 275: Regurgitation
		# 276: Coryza
		# 277: Pain in eye
		# 278: Vomiting
		# 279: Fluid retention
		# 280: Hot flashes
		# 281: Upper abdominal pain
		# 282: Skin rash
		# 283: Rib pain
		# 284: Swollen abdomen
		# 285: Infant feeding problem
		# 286: Knee pain
		# 287: Redness in ear
		# 288: Flatulence
		# 289: Chills
		# 290: Depression
		# 291: Weight gain
		# 292: Muscle pain
		# 293: Fluid in ear
		# 294: Penis pain
		# 295: Abusing alcohol
		# 296: Retention of urine
		# 297: Drainage in throat
		# 298: Skin irritation
		# 299: Shortness of breath

		}
		return option[iden]



