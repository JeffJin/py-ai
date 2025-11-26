import numpy as np
import pandas as pd

# Format: (string, label). +1 is valid, -1 is invalid.
test_emails = [
    ("jeff@gmail.com"),  # Perfect
    ("admin@site.org"),  # Perfect
    ("info@co.uk"),  # Perfect
    ("jeffgmail.com"),  # Missing @
    ("jeff@@gmail.com"),  # double @
    ("jeff@@gmail.c"),  # double @
    ("jeff@gmailcom"),  # Missing .
    ("a@b"),  # Too short
    ("hello"),  # Missing everything
]
dtype_spec = {
    "email": "string",
    "is_valid": "int64"
}

training_data = pd.read_csv(
    "data/emails1.csv",
    dtype=dtype_spec
)

def get_email_feature_templates(email):
    email = str(email)
    # The dictionary acts as our Sparse Vector
    # keys = feature names, values = feature values
    phi = {}

    # --- Template 1: Suffix (from your image) ---
    # "Last three characters equals ____"
    # This automatically creates features like 'suffix=com', 'suffix=io', 'suffix=xyz'
    if len(email) >= 2:
        suffix = email[-2:]
        phi[f"suffix={suffix}"] = 1

    if len(email) >= 3:
        suffix = email[-3:]
        phi[f"suffix={suffix}"] = 1

    if len(email) >= 4:
        suffix = email[-4:]
        phi[f"suffix={suffix}"] = 1

    # --- Template 2: Length Threshold (from your image) ---
    # "Length greater than ____"
    # We can add multiple thresholds automatically
    phi["len>5"] = 1 if len(email) > 5 else 0
    phi["len>10"] = 1 if len(email) > 10 else 0

    # --- Template 3: Alpha Fraction (from your image) ---
    # "Fraction of alphanumeric characters"
    # This detects weird symbols. 'abc@123' is high, '##$$@!!' is low.
    alpha_num_count = sum(c.isalnum() for c in email)
    phi["frac_alnum"] = alpha_num_count / len(email) if len(email) > 0 else 0

    # --- Template 4: Character Counts (The Fix for your '@' issue) ---
    # Instead of checking IF it exists, we count it.
    # We create a feature specifically for the count.
    at_count = email.count('@')
    phi[f"count_@={at_count}"] = 1

    # Template 5: Character Counts (The Fix for your '.' issue)
    # We can do the same for dots
    dot_count = email.count('.')
    phi[f"count_.={dot_count}"] = 1

    return phi

def initial_weights():
    return {}

def sigmoid(z):
    # np.exp is the efficient numpy version of e^x
    # Prevent overflow for very large negative/positive z
    if z > 15: return 1.0
    if z < -15: return 0.0
    return 1 / (1 + np.exp(-z))

# Sparse Dot Product (The "Score" Calculator)
def sparse_dot_product(features, w):
    return sum(value * w.get(key, 0.0) for key, value in features.items())

# Dictionary x scalar value (The gradient "-phi(x) * y" Calculator)
def dict_product(features, y):
    for key, value in features.items():
        features[key] = value * y
    return features

def dict_add(features, val):
    for key, value in features.items():
        features[key] = value + val.get(key, 0.0)
    return features

def dict_subtract(features, val):
    for key, value in features.items():
        features[key] = value - val.get(key, 0.0)
    return features

def hinge_loss(w, x, y):
    phi = get_email_feature_templates(x)
    margin = sparse_dot_product(w, phi) * y
    print(f'weights = {w}, x={x}, y={y}, phi(x)={phi}, margin={margin}, raw_loss={1 - margin}')
    return max(1 - margin, 0)

def logistic_loss(w, x, y):
    phi = get_email_feature_templates(x)
    local_score = sparse_dot_product(w, phi)
    return np.log(1 + np.exp(-local_score * y))

# --------------------------Option 1-------------------------
def train_logistic_sgd(email, y_raw, w, learning_rate, lambda_reg=0.0):
    """
    email: The input string
    y_raw: +1 for Valid, -1 for Invalid
    w: The weights dictionary
    """

    # A. Setup Target (Convert -1/+1 to 0/1 for probability math)
    target = 1 if y_raw == 1 else 0

    # B. Generate Features
    features = get_email_feature_templates(email)

    # C. Forward Pass (Calculate Prediction)
    z = sparse_dot_product(features, w)
    prediction = sigmoid(z)

    # D. Calculate Error Term (scalar)
    # Gradient of Log Loss wrt z is simply (prediction - target)
    error_term = prediction - target

    # E. Update Weights (Sparse SGD)
    for key, x_val in features.items():
        # 1. Lazy Initialization: If weight doesn't exist, start at 0
        if key not in w:
            w[key] = 0.0

        # 2. Calculate Gradient for this specific weight
        # Gradient = (pred - y) * x + (2 * lambda * w)
        # Note: We apply regularization only to features present in this example
        # (This is an approximation called "Lazy Regularization")
        gradient = (error_term * x_val) + (2 * lambda_reg * w[key])

        # 3. Descent Step
        w[key] = w[key] - (learning_rate * gradient)

    return prediction, w

def train_batch_logistic_sgd(data):
    weights = initial_weights()
    learning_rate = 0.1
    for _, row in data.iterrows():
        email, label = row['email'], row['label']
        pred, weights = train_logistic_sgd(email, label, weights, learning_rate)

        # Formatting for display
        target_display = "1 (Valid)" if label == 1 else "0 (Invalid)"
        print(f"{email:<15} | {target_display:<6} | {pred:.4f} ({pred * 100:.1f}%)   | Updated w")
    return weights
# --------------------------End of Option 1-------------------------

# --------------------------Option 2-------------------------
lambda_param = 0.01
def train_one_example(email, y_true, w):
    # A. Generate Features (using the template function from before)
    # e.g., {'suffix=com': 1, 'len>5': 1}
    features = get_email_feature_templates(email)

    # B. Predict
    local_score = sparse_dot_product(features, w)

    # C. Check for Error (Perceptron/Hinge logic)
    # If margin is violated...
    if y_true * local_score <= 0:

        # D. Update Weights
        for key, value in features.items():
            # ERROR CORRECTION:
            # w[new] = w[old] + (learning_rate * y * x)

            # 1. If this is the first time seeing this feature, "birth" it.
            if key not in w:
                w[key] = 0.0

            # 2. Update it
            w[key] += lambda_param * y_true * value
    return w

def train_batch_examples(data):
    w = initial_weights()
    for _, row in data.iterrows():
        w = train_one_example(row['email'], row['label'], w)
    return w
# --------------------------End of Option 2-------------------------

# --------------------------Option 3-------------------------
def train_loss(w):
    return 1 / len(training_data) * sum(hinge_loss(w, row['email'], row['label']) for _, row in training_data.iterrows())

def train_svm_sgd(w, email, label, step = 0.01):
    phi = get_email_feature_templates(email)
    margin = sparse_dot_product(w, phi) * label
    if 1 > margin:
        gradient = {}
        for key, y_val in phi.items():
            # 1. Lazy Initialization: If weight doesn't exist, start at 0
            if key not in w:
                w[key] = 0.0
            # 2. Calculate Gradient for this specific weight
            gradient[key] = -label * phi[key]
        # update weights based on gradient and step size
        for k, g in list(gradient.items()):
            w[k] = w.get(k, 0) - step * g
    return w

def train_svm_batch(step):
    w = initial_weights()
    iteration = 0  # Manual iteration counter

    weights = {}
    # ### START CODE HERE ###
    for epoch in range(50):
        for _, row in training_data.iterrows():
            weights = train_svm_sgd(weights, row['email'], row['label'], step)

    return weights
# --------------------------End of Option 3-------------------------
# Train the model
# WEIGHTS = train_batch_logistic_sgd(training_data)
# WEIGHTS = train_batch_examples(training_data)
WEIGHTS = train_svm_batch(0.01)
# w = {'suffix=ai': -0.03, 'suffix=.ai': -0.019999999999999997, 'suffix=s.ai': 0.020000000000000004, 'len>5': -0.18000000000000002, 'len>10': -0.18000000000000002, 'frac_alnum': 0.40365006605764187, 'count_@=1': 0.03, 'count_.=1': 0.03, 'suffix=ki': -0.019999999999999997, 'suffix=iki': -0.019999999999999997, 'suffix=wiki': -0.019999999999999997, 'count_.=0': -0.09, 'suffix=et': -0.019999999999999997, 'suffix=net': -0.019999999999999997, 'suffix=.net': -0.010000000000000004, 'count_@=2': -0.08, 'suffix=uk': 0.01, 'suffix=.uk': 0.019999999999999997, 'suffix=o.uk': 0.05, 'count_.=2': -0.060000000000000005, 'suffix=au': 0.009999999999999997, 'suffix=.au': 0.019999999999999997, 'suffix=m.au': 0.05, 'suffix=fo': 3.469446951953614e-18, 'suffix=nfo': 3.469446951953614e-18, 'suffix=info': 3.469446951953614e-18, 'suffix=..au': -0.03, 'count_.=4': -0.060000000000000005, 'suffix=me': -0.019999999999999997, 'suffix=.me': -0.009999999999999997, 'suffix=a.me': 0.0, 'count_@=0': -0.12999999999999998, 'suffix=co': -0.019999999999999997, 'suffix=.co': -0.019999999999999997, 'suffix=r.co': 0.02, 'suffix=io': -0.03, 'suffix=.io': -0.009999999999999997, 'suffix=n.io': 0.009999999999999997, 'suffix=om': -0.019999999999999997, 'suffix=com': -0.019999999999999997, 'suffix=.com': -0.009999999999999997, 'suffix=y.me': 0.03, 'suffix=du': 3.469446951953614e-18, 'suffix=edu': 3.469446951953614e-18, 'suffix=.edu': 3.469446951953614e-18, 'suffix=og': -0.019999999999999997, 'suffix=log': -0.019999999999999997, 'suffix=blog': -0.019999999999999997, 'suffix=ame': -0.009999999999999997, 'suffix=name': -0.009999999999999997, 'suffix=s.me': -0.019999999999999997, 'suffix=cy': -0.009999999999999997, 'suffix=ncy': -0.009999999999999997, 'suffix=ency': -0.009999999999999997, 'suffix=a.co': -0.02, 'suffix=s.io': 0.019999999999999997, 'suffix=e.co': -0.02, 'suffix=rg': -0.009999999999999997, 'suffix=org': -0.009999999999999997, 'suffix=.org': -0.009999999999999997, 'suffix=y.ai': 0.0, 'suffix=nnet': -0.01, 'suffix=r.me': 3.469446951953614e-18, 'suffix=n.me': 3.469446951953614e-18, 'suffix=g.io': 0.01, 'suffix=n.co': -3.469446951953614e-18, 'suffix=lio': -0.01, 'suffix=llio': -0.01, 'suffix=o.ai': 0.0, 'suffix=t.co': -0.01, 'suffix=y.io': 0.019999999999999997, 'suffix=d.ai': -0.02, 'suffix=t.io': -0.01, 'suffix=d.me': -0.009999999999999997, 'suffix=..ai': -0.01, 'suffix=z.co': 0.04, 'suffix=..uk': -0.03, 'suffix=z.io': -0.02, 'suffix=r.io': 3.469446951953614e-18, 'suffix=s.co': -0.010000000000000004, 'suffix=t.me': 0.0, 'suffix=k.ai': -0.01, 'suffix=e.ai': 0.01, 'suffix=m.ai': -0.01, 'suffix=l.co': -0.019999999999999997, 'suffix=z.me': -0.03, 'suffix=..io': -0.01, 'suffix=a.ai': 0.009999999999999997, 'suffix=z.ai': 3.469446951953614e-18, 'suffix=e.io': -0.019999999999999997, 'suffix=n.ai': -3.469446951953614e-18, 'suffix=g.co': -0.01, 'suffix=o.io': -0.009999999999999997, 'suffix=x.co': 0.0, 'suffix=y.co': 0.02, 'suffix=l.ai': 0.009999999999999997, 'suffix=t.ai': -0.01, 'suffix=d.io': -0.019999999999999997, 'suffix=ouk': -0.01, 'suffix=couk': -0.01, 'suffix=l.me': 0.0, 'suffix=o.co': -0.03, 'suffix=k.me': 0.0, 'suffix=l.io': -0.01, 'suffix=k.io': 0.009999999999999997, 'suffix=m.io': 0.0, 'suffix=d.co': 0.02, 'suffix=h.ai': -0.01, 'suffix=r.ai': 0.0, 'suffix=b.ai': -0.01, 'suffix=h.co': -0.01, 'suffix=h.io': 0.01, 'suffix=p.co': 0.0, 'suffix=a.io': -3.469446951953614e-18, 'suffix=p.me': 0.0, 'suffix=mau': -0.01, 'suffix=omau': -0.01, 'suffix=w.me': -0.01, 'suffix=h.me': 0.02, 'suffix=k.co': 0.02, 'suffix=o.me': 0.0, 'suffix=w.ai': 0.01, 'suffix=g.ai': -0.01, 'suffix=scom': -0.01, 'suffix=u.io': 0.01, 'suffix=x.me': 0.01, 'suffix=rio': -0.01, 'suffix=erio': -0.01, 'suffix=w.co': -0.01, 'suffix=e.me': 0.01, 'suffix=dai': -0.01, 'suffix=ndai': -0.01, 'suffix=g.me': -0.01, 'suffix=u.ai': 0.01}
# w = {'suffix=ai': np.float64(-0.40930459472678693), 'suffix=.ai': np.float64(-0.2784353766943942), 'suffix=s.ai': np.float64(-0.29296526685725033), 'len>5': np.float64(-3.3717255865765923), 'len>10': np.float64(-3.3717255865765923), 'frac_alnum': np.float64(3.597491328473916), 'count_@=1': np.float64(2.6417406517628086), 'count_.=1': np.float64(2.8601315715984925), 'suffix=ki': np.float64(-0.5742633219571006), 'suffix=iki': np.float64(-0.5742633219571006), 'suffix=wiki': np.float64(-0.5742633219571006), 'count_.=0': np.float64(-3.014395252915496), 'suffix=og': np.float64(-0.29978427392689205), 'suffix=log': np.float64(-0.29978427392689205), 'suffix=blog': np.float64(-0.29978427392689205), 'suffix=cy': np.float64(-0.10594778100041118), 'suffix=ncy': np.float64(-0.10594778100041118), 'suffix=ency': np.float64(-0.10594778100041118), 'suffix=et': np.float64(-0.6521998514488592), 'suffix=net': np.float64(-0.6521998514488592), 'suffix=.net': np.float64(-0.5033196475441445), 'count_@=2': np.float64(-2.9260431777624745), 'suffix=uk': np.float64(0.8566308435798612), 'suffix=.uk': np.float64(1.126919706418438), 'suffix=o.uk': np.float64(2.18258762825319), 'count_.=2': np.float64(-1.2690679506230735), 'suffix=fo': np.float64(-0.2202335898911852), 'suffix=nfo': np.float64(-0.2202335898911852), 'suffix=info': np.float64(-0.2202335898911852), 'suffix=om': np.float64(-0.3870774476504375), 'suffix=com': np.float64(-0.3870774476504375), 'suffix=.com': np.float64(-0.20847581767400702), 'suffix=au': np.float64(0.798276293014203), 'suffix=.au': np.float64(1.0367847053040384), 'suffix=m.au': np.float64(1.9295107381057877), 'suffix=..au': np.float64(-0.8927260328017491), 'count_.=4': np.float64(-1.9483939546364957), 'suffix=me': np.float64(-0.5801821567038652), 'suffix=.me': np.float64(-0.18608229308511376), 'suffix=a.me': np.float64(-0.24776957626350674), 'count_@=0': np.float64(-3.087423060576908), 'suffix=co': np.float64(-0.45363672320237636), 'suffix=.co': np.float64(-0.2711400670206303), 'suffix=r.co': np.float64(0.051425137413257535), 'suffix=io': np.float64(-0.6479040111807278), 'suffix=.io': np.float64(-0.4056340201476857), 'suffix=n.io': np.float64(0.2934667352115907), 'suffix=y.me': np.float64(0.44915900293441846), 'suffix=ame': np.float64(-0.21489341027927092), 'suffix=name': np.float64(-0.21489341027927092), 'suffix=rg': np.float64(-0.4131979829666972), 'suffix=org': np.float64(-0.4131979829666972), 'suffix=.org': np.float64(-0.23455362794499057), 'suffix=du': np.float64(-0.28290098851530127), 'suffix=edu': np.float64(-0.28290098851530127), 'suffix=.edu': np.float64(-0.11137252679094922), 'suffix=ouk': np.float64(-0.27028886283857756), 'suffix=couk': np.float64(-0.27028886283857756), 'suffix=s.me': np.float64(-0.2464686693335421), 'suffix=a.co': np.float64(-0.12123050977377768), 'suffix=n.me': np.float64(0.15357528291537123), 'suffix=s.io': np.float64(-0.010760447840838409), 'suffix=e.co': np.float64(-0.23408487135337125), 'suffix=n.ai': np.float64(0.2489436144464312), 'suffix=n.co': np.float64(0.24831043029766603), 'suffix=g.io': np.float64(0.16060100414308276), 'suffix=lio': np.float64(-0.09657657311011202), 'suffix=llio': np.float64(-0.09140137945663043), 'suffix=a.ai': np.float64(0.08898379721715982), 'suffix=d.io': np.float64(-0.2665177952908522), 'suffix=y.ai': np.float64(0.14072870345011448), 'suffix=ome': np.float64(-0.05626660018277591), 'suffix=come': np.float64(-0.03462726043532031), 'suffix=zcom': np.float64(-0.05241353559454662), 'suffix=nnet': np.float64(-0.08607084937022015), 'suffix=r.me': np.float64(-0.009024477260387149), 'suffix=e.me': np.float64(0.24255614629663647), 'suffix=r.ai': np.float64(0.1271747767501779), 'suffix=..uk': np.float64(-1.0556679218347471), 'suffix=r.io': np.float64(0.04392695289757205), 'suffix=z.co': np.float64(0.49164627401290384), 'suffix=e.ai': np.float64(0.11849563084343147), 'suffix=o.ai': np.float64(-0.06604648454452824), 'suffix=..io': np.float64(-0.3699160420468517), 'suffix=t.co': np.float64(0.04892600927338426), 'suffix=y.io': np.float64(0.22771259110914716), 'suffix=o.co': np.float64(-0.23057960021604396), 'suffix=d.ai': np.float64(-0.14504563245171484), 'suffix=g.me': np.float64(0.027941424707691828), 'suffix=korg': np.float64(-0.06043998304112251), 'suffix=redu': np.float64(-0.05164166977532418), 'suffix=t.io': np.float64(-0.03686424851963929), 'suffix=d.me': np.float64(-0.32134156729731), 'suffix=s.co': np.float64(0.10980052781189972), 'suffix=..ai': np.float64(-0.38008577959600265), 'suffix=h.me': np.float64(0.1455667664508633), 'suffix=zco': np.float64(-0.026525452459907582), 'suffix=ezco': np.float64(-0.026525452459907582), 'suffix=z.io': np.float64(-0.015971248684216385), 'suffix=yco': np.float64(-0.030074526741744247), 'suffix=eyco': np.float64(-0.02242625911041687), 'suffix=mau': np.float64(-0.2385084122898351), 'suffix=omau': np.float64(-0.2385084122898351), 'suffix=x.io': np.float64(0.052591657283838814), 'suffix=h.co': np.float64(-0.10904401177312557), 'suffix=eco': np.float64(-0.020386900494252985), 'suffix=neco': np.float64(-0.015546635696835877), 'suffix=sio': np.float64(-0.029780361798118306), 'suffix=nsio': np.float64(-0.02349141567387151), 'suffix=rio': np.float64(-0.05100951374884656), 'suffix=rrio': np.float64(-0.010526807443191115), 'suffix=t.me': np.float64(0.12184788550793152), 'suffix=k.ai': np.float64(-0.127708230074825), 'suffix=t.ai': np.float64(0.12590525855702264), 'suffix=..co': np.float64(-0.23492001787312947), 'suffix=e.io': np.float64(-0.23991648248718436), 'suffix=k.io': np.float64(0.011409427322854857), 'suffix=eme': np.float64(-0.035198200210995643), 'suffix=teme': np.float64(-0.02862275644251569), 'suffix=erio': np.float64(-0.03967944860099426), 'suffix=m.ai': np.float64(-0.006171291756036165), 'suffix=l.co': np.float64(-0.22491980946827708), 'suffix=z.me': np.float64(-0.2948701459438511), 'suffix=wai': np.float64(-0.012731964572676967), 'suffix=awai': np.float64(-0.012731964572676967), 'suffix=z.ai': np.float64(0.02608108799558313), 'suffix=h.io': np.float64(0.09340899534813343), 'suffix=b.co': np.float64(0.031498095216721624), 'suffix=a.io': np.float64(-0.05947533098225079), 'suffix=g.co': np.float64(-0.23773467813806642), 'suffix=nco': np.float64(-0.03108479836738957), 'suffix=onco': np.float64(-0.020855522251981836), 'suffix=..me': np.float64(-0.30758811077044856), 'suffix=yio': np.float64(-0.010787223821634615), 'suffix=eyio': np.float64(-0.010787223821634615), 'suffix=nome': np.float64(-0.018522984287968322), 'suffix=ecom': np.float64(-0.025370381341180158), 'suffix=tme': np.float64(-0.024018568290835908), 'suffix=htme': np.float64(-0.01827828383819117), 'suffix=o.io': np.float64(-0.11272974176379771), 'suffix=x.co': np.float64(0.08636313296205084), 'suffix=nedu': np.float64(-0.03254935042932224), 'suffix=eai': np.float64(-0.015181939378205807), 'suffix=seai': np.float64(-0.010730540936945398), 'suffix=y.co': np.float64(0.03140851472168558), 'suffix=l.ai': np.float64(0.156582643836534), 'suffix=sco': np.float64(-0.03710996177304209), 'suffix=lsco': np.float64(-0.008714089948475284), 'suffix=aorg': np.float64(-0.02948878407928295), 'suffix=nai': np.float64(-0.021713276149433875), 'suffix=onai': np.float64(-0.012192720241840307), 'suffix=snet': np.float64(-0.015452129238516804), 'suffix=scom': np.float64(-0.04691725908290556), 'suffix=b.me': np.float64(0.09420131762413583), 'suffix=dedu': np.float64(-0.01294857140817664), 'suffix=yedu': np.float64(-0.009089334743412282), 'suffix=l.me': np.float64(-0.10177745475835726), 'suffix=isco': np.float64(-0.010594060723455522), 'suffix=w.ai': np.float64(0.08229572636690324), 'suffix=sorg': np.float64(-0.024883592833765233), 'suffix=i.ai': np.float64(-0.011626138667819325), 'suffix=o.me': np.float64(0.02555706240299551), 'suffix=tco': np.float64(-0.012677819036019061), 'suffix=ntco': np.float64(-0.009654534008313556), 'suffix=norg': np.float64(-0.03375917855679182), 'suffix=k.co': np.float64(0.10041689522628176), 'suffix=hio': np.float64(-0.010005434247922558), 'suffix=thio': np.float64(-0.00844981367995693), 'suffix=zorg': np.float64(-0.011570180864809813), 'suffix=sai': np.float64(-0.04074069108537029), 'suffix=isai': np.float64(-0.011264586485275983), 'suffix=h.ai': np.float64(-0.1817765471408571), 'suffix=k.me': np.float64(0.03490236520219399), 'suffix=nio': np.float64(-0.017095813000194363), 'suffix=anio': np.float64(-0.009959328003133084), 'suffix=l.io': np.float64(-0.25831647286227644), 'suffix=dcom': np.float64(-0.008155155344335556), 'suffix=rsco': np.float64(-0.004930860171991085), 'suffix=osai': np.float64(-0.007108259588798262), 'suffix=aco': np.float64(-0.0051789121669044485), 'suffix=raco': np.float64(-0.004213900221202909), 'suffix=ssco': np.float64(-0.0027944884823276205), 'suffix=enet': np.float64(-0.015895711869006904), 'suffix=lco': np.float64(-0.0056791849292602255), 'suffix=llco': np.float64(-0.0056791849292602255), 'suffix=zio': np.float64(-0.007616674098388413), 'suffix=ezio': np.float64(-0.007616674098388413), 'suffix=d.co': np.float64(-0.0590831456117355), 'suffix=eedu': np.float64(-0.011512662217568505), 'suffix=alio': np.float64(-0.005175193653481589), 'suffix=m.io': np.float64(-0.035518195024192595), 'suffix=nme': np.float64(-0.025356027138670353), 'suffix=onme': np.float64(-0.015595051796485638), 'suffix=kcom': np.float64(-0.0069384418491457035), 'suffix=ksai': np.float64(-0.006086838990172745), 'suffix=p.ai': np.float64(-0.07960146305506917), 'suffix=sedu': np.float64(-0.020561493085200905), 'suffix=b.ai': np.float64(-0.008607852920443783), 'suffix=lcom': np.float64(-0.005671518933823641), 'suffix=hnet': np.float64(-0.008088359100145222), 'suffix=tai': np.float64(-0.0026372836730593876), 'suffix=ntai': np.float64(-0.0018818058928457054), 'suffix=zai': np.float64(-0.009375608268598496), 'suffix=ezai': np.float64(-0.006732935374407164), 'suffix=ncom': np.float64(-0.011192437431770768), 'suffix=rme': np.float64(-0.01068102999321735), 'suffix=orme': np.float64(-0.002280558232963848), 'suffix=g.ai': np.float64(-0.152587947199342), 'suffix=p.co': np.float64(-0.02326015768836953), 'suffix=inme': np.float64(-0.00468955063818407), 'suffix=tzai': np.float64(-0.002642672894191332), 'suffix=ynet': np.float64(-0.006685023257928871), 'suffix=enco': np.float64(-0.006719286232244915), 'suffix=p.io': np.float64(0.03502714507387266), 'suffix=pio': np.float64(-0.004700678493007075), 'suffix=ipio': np.float64(-0.0035498803630132922), 'suffix=sme': np.float64(-0.012846118140890494), 'suffix=isme': np.float64(-0.005089360512843892), 'suffix=aedu': np.float64(-0.0032569209950521574), 'suffix=tnet': np.float64(-0.002168756318735484), 'suffix=iai': np.float64(-0.0017672161654983564), 'suffix=liai': np.float64(-0.0017672161654983564), 'suffix=anme': np.float64(-0.0025716353005227357), 'suffix=msme': np.float64(-0.00595267626945808), 'suffix=p.me': np.float64(-0.01705634872965467), 'suffix=nsco': np.float64(-0.004282684570591698), 'suffix=gai': np.float64(-0.003546450681257871), 'suffix=ngai': np.float64(-0.003546450681257871), 'suffix=esai': np.float64(-0.00916249074889305), 'suffix=rsai': np.float64(-0.003560496764631423), 'suffix=keme': np.float64(-0.004650782917234092), 'suffix=w.me': np.float64(-0.0785830280786289), 'suffix=zedu': np.float64(-0.013469525029663115), 'suffix=aai': np.float64(-0.0015788409702618266), 'suffix=laai': np.float64(-0.0015788409702618266), 'suffix=erme': np.float64(-0.008400471760253502), 'suffix=rorg': np.float64(-0.007078516714360986), 'suffix=enai': np.float64(-0.002650484684265816), 'suffix=rai': np.float64(-0.00748349259261238), 'suffix=erai': np.float64(-0.00748349259261238), 'suffix=leco': np.float64(-0.0009601573507427967), 'suffix=enio': np.float64(-0.00104678571263504), 'suffix=medu': np.float64(-0.0018563205397819122), 'suffix=oio': np.float64(-0.00501353146292351), 'suffix=loio': np.float64(-0.00501353146292351), 'suffix=onet': np.float64(-0.00255433734178342), 'suffix=tio': np.float64(-0.0026238969301917263), 'suffix=utio': np.float64(-0.001645482964015074), 'suffix=ryco': np.float64(-0.0038225126977958048), 'suffix=hcom': np.float64(-0.0025132917062758416), 'suffix=hedu': np.float64(-0.002142406189613886), 'suffix=m.co': np.float64(0.06494067398860089), 'suffix=x.me': np.float64(0.10249292288786385), 'suffix=hai': np.float64(-0.003092809672993731), 'suffix=thai': np.float64(-0.003092809672993731), 'suffix=ntme': np.float64(-0.003436169382404644), 'suffix=yorg': np.float64(-0.004350556354434917), 'suffix=u.co': np.float64(0.019048477478266004), 'suffix=ycom': np.float64(-0.00776378024105857), 'suffix=onio': np.float64(-0.00608969928442624), 'suffix=mai': np.float64(-0.0019442733355328767), 'suffix=amai': np.float64(-0.0019442733355328767), 'suffix=rco': np.float64(-0.009129426056921837), 'suffix=erco': np.float64(-0.0076096995287740065), 'suffix=msai': np.float64(-0.0026550342241919837), 'suffix=hme': np.float64(-0.004123748167203084), 'suffix=ghme': np.float64(-0.0015464689715976978), 'suffix=rtco': np.float64(-0.0010510079840515233), 'suffix=anai': np.float64(-0.005521237010595878), 'suffix=gio': np.float64(-0.002468406805512912), 'suffix=ngio': np.float64(-0.002468406805512912), 'suffix=kedu': np.float64(-0.004651499079211554), 'suffix=tedu': np.float64(-0.002437368355624965), 'suffix=isio': np.float64(-0.0016543259304261744), 'suffix=wnco': np.float64(-0.0014381108603003694), 'suffix=dyco': np.float64(-0.0021411752111691047), 'suffix=m.me': np.float64(0.040596908420470215), 'suffix=some': np.float64(-0.0012559379298129755), 'suffix=neai': np.float64(-0.001323075445816885), 'suffix=dorg': np.float64(-0.000446516324171657), 'suffix=yme': np.float64(-0.0031019297504588667), 'suffix=ryme': np.float64(-0.0016981332203921775), 'suffix=yeai': np.float64(-0.0016418791333938248), 'suffix=hco': np.float64(-0.0018161406167192735), 'suffix=thco': np.float64(-0.0018161406167192735), 'suffix=u.io': np.float64(0.04718832728814958), 'suffix=osco': np.float64(-0.0014261471197047797), 'suffix=wsco': np.float64(-0.0013054879313411146), 'suffix=znet': np.float64(-0.0031865622010824463), 'suffix=thme': np.float64(-0.002577279195605386), 'suffix=rome': np.float64(-0.0018604175296743112), 'suffix=hyme': np.float64(-0.001403796530066689), 'suffix=asio': np.float64(-0.0031680450582490752), 'suffix=b.io': np.float64(0.0367016493195237), 'suffix=ocom': np.float64(-0.004179432428804743), 'suffix=anet': np.float64(-0.002158694199153163), 'suffix=oedu': np.float64(-0.0030503610631104944), 'suffix=nsme': np.float64(-0.0008632203349361756), 'suffix=lai': np.float64(-0.0008949235178957934), 'suffix=llai': np.float64(-0.0008949235178957934), 'suffix=teco': np.float64(-0.000831902253052449), 'suffix=esco': np.float64(-0.0016307126814263295), 'suffix=rtme': np.float64(-0.0023041150702400952), 'suffix=yai': np.float64(-0.002556050914287891), 'suffix=hyai': np.float64(-0.00126548842828391), 'suffix=byco': np.float64(-0.0016845797223624635), 'suffix=mcom': np.float64(-0.002229579378818047), 'suffix=zme': np.float64(-0.0021244504767855307), 'suffix=ezme': np.float64(-0.0021244504767855307), 'suffix=dme': np.float64(-0.0037457615306076743), 'suffix=idme': np.float64(-0.002661985655359987), 'suffix=rnet': np.float64(-0.003944814481122628), 'suffix=rcom': np.float64(-0.003306400982388432), 'suffix=ltco': np.float64(-0.0019722770436539807), 'suffix=esio': np.float64(-0.0006234781113236489), 'suffix=rtai': np.float64(-0.0007554777802136823), 'suffix=aio': np.float64(-0.0021657695565948437), 'suffix=laio': np.float64(-0.0012095402976123066), 'suffix=rpio': np.float64(-0.0011507981299937826), 'suffix=ume': np.float64(-0.0009220143211135636), 'suffix=wume': np.float64(-0.0009220143211135636), 'suffix=eorg': np.float64(-0.0021907919775385077), 'suffix=wnme': np.float64(-0.0013501286636796875), 'suffix=w.co': np.float64(-0.08006743352745127), 'suffix=oorg': np.float64(-0.0018461639144829298), 'suffix=eeme': np.float64(-0.000924290718464402), 'suffix=eeco': np.float64(-0.0030482051936218646), 'suffix=bedu': np.float64(-0.000804157462520631), 'suffix=tcom': np.float64(-0.001950415661377062), 'suffix=unet': np.float64(-0.0015092866836753378), 'suffix=arco': np.float64(-0.0015197265281478296), 'suffix=mco': np.float64(-0.00214236685885043), 'suffix=amco': np.float64(-0.00214236685885043), 'suffix=nnai': np.float64(-0.0008362643035802413), 'suffix=asco': np.float64(-0.0014314301437286594), 'suffix=ssio': np.float64(-0.0008430970242479032), 'suffix=oeme': np.float64(-0.0010003701327814616), 'suffix=reai': np.float64(-0.001486443862049699), 'suffix=dai': np.float64(-0.005624397054707056), 'suffix=idai': np.float64(-0.0030935650414190465), 'suffix=lorg': np.float64(-0.0013642604685032617), 'suffix=ndai': np.float64(-0.00253083201328801), 'suffix=gedu': np.float64(-0.0002873987062643706), 'suffix=oco': np.float64(-0.0006911666807341691), 'suffix=noco': np.float64(-0.0006911666807341691), 'suffix=lsme': np.float64(-0.0009408610236523462), 'suffix=dtio': np.float64(-0.0009784139661766524), 'suffix=saio': np.float64(-0.000956229258982537), 'suffix=laco': np.float64(-0.0009650119457015396), 'suffix=inco': np.float64(-0.0020718790228624515), 'suffix=orio': np.float64(-0.0008032577046611913), 'suffix=lme': np.float64(-0.0008220051359245219), 'suffix=elme': np.float64(-0.0008220051359245219), 'suffix=eio': np.float64(-0.000584420760566836), 'suffix=veio': np.float64(-0.000584420760566836), 'suffix=wedu': np.float64(-0.0012694226445038447), 'suffix=nsai': np.float64(-0.0009029842834068433), 'suffix=chio': np.float64(-0.0015556205679656291), 'suffix=torg': np.float64(-0.001225829892441666), 'suffix=rdme': np.float64(-0.0010837758752476872), 'suffix=dnet': np.float64(-0.0011656798433443414), 'suffix=f.io': np.float64(-0.0016824996433509678), 'suffix=inai': np.float64(-0.0005125699091516359), 'suffix=oyai': np.float64(-0.0012905624860039809), 'suffix=enme': np.float64(-0.0011496607397982201), 'suffix=u.ai': np.float64(0.03697313245496682), 'suffix=x.ai': np.float64(0.02162288565116982), 'suffix=dio': np.float64(-0.0018416931990291605), 'suffix=rdio': np.float64(-0.0018416931990291605)}
# w = {'suffix=uk': 16.89999999999997, 'suffix=.uk': 14.599999999999964, 'suffix=o.uk': 11.299999999999976, 'len>5': 252.69999999998993, 'len>10': 252.69999999998993, 'frac_alnum': 194.3913468436705, 'count_@=1': 182.9999999999939, 'count_.=2': 54.4000000000005, 'suffix=me': 32.800000000000196, 'suffix=ame': 18.499999999999993, 'suffix=name': 18.499999999999993, 'count_.=1': 154.6999999999955, 'suffix=du': 16.99999999999997, 'suffix=edu': 16.99999999999997, 'suffix=dedu': 0.5, 'count_.=0': 38.400000000000276, 'suffix=.me': 12.099999999999973, 'suffix=..me': 1.7000000000000004, 'suffix=hme': 0.2, 'suffix=ghme': 0.1, 'suffix=et': 17.199999999999974, 'suffix=net': 17.199999999999974, 'suffix=.net': 14.999999999999963, 'count_@=0': 35.600000000000236, 'suffix=nnet': 0.8999999999999999, 'suffix=au': 15.499999999999961, 'suffix=.au': 13.799999999999967, 'suffix=m.au': 11.899999999999974, 'count_@=2': 34.100000000000215, 'suffix=io': 19.500000000000007, 'suffix=.io': 16.99999999999997, 'suffix=d.io': 1.0999999999999999, 'suffix=cy': 14.499999999999964, 'suffix=ncy': 14.499999999999964, 'suffix=ency': 14.499999999999964, 'suffix=..uk': 3.3000000000000016, 'count_.=4': 5.1999999999999975, 'suffix=co': 15.199999999999962, 'suffix=.co': 12.299999999999972, 'suffix=z.co': 0.8999999999999999, 'suffix=.edu': 12.99999999999997, 'suffix=ki': 20.30000000000002, 'suffix=iki': 20.30000000000002, 'suffix=wiki': 20.30000000000002, 'suffix=og': 13.399999999999968, 'suffix=log': 13.399999999999968, 'suffix=blog': 13.399999999999968, 'suffix=..io': 3.600000000000002, 'suffix=lio': 0.30000000000000004, 'suffix=llio': 0.2, 'suffix=fo': 21.300000000000033, 'suffix=nfo': 21.300000000000033, 'suffix=info': 21.300000000000033, 'suffix=rg': 17.399999999999977, 'suffix=org': 17.399999999999977, 'suffix=.org': 15.099999999999962, 'suffix=ai': 16.99999999999997, 'suffix=.ai': 13.699999999999967, 'suffix=n.ai': 2.800000000000001, 'suffix=y.ai': 0.5, 'suffix=y.io': 1.0999999999999999, 'suffix=korg': 0.4, 'suffix=n.co': 2.800000000000001, 'suffix=norg': 0.4, 'suffix=a.io': 0.30000000000000004, 'suffix=d.ai': 0.8999999999999999, 'suffix=d.co': 0.5, 'suffix=ouk': 2.3000000000000007, 'suffix=couk': 2.3000000000000007, 'suffix=z.io': 0.7, 'suffix=eai': 0.2, 'suffix=seai': 0.1, 'suffix=aorg': 0.30000000000000004, 'suffix=sai': 1.0999999999999999, 'suffix=isai': 0.1, 'suffix=o.ai': 0.2, 'suffix=..ai': 2.3000000000000007, 'suffix=mau': 1.7000000000000004, 'suffix=omau': 1.7000000000000004, 'suffix=z.me': 0.7999999999999999, 'suffix=r.io': 2.0000000000000004, 'suffix=e.me': 0.30000000000000004, 'suffix=r.co': 0.9999999999999999, 'suffix=s.ai': 2.0000000000000004, 'suffix=n.me': 2.600000000000001, 'suffix=om': 14.699999999999964, 'suffix=com': 14.699999999999964, 'suffix=.com': 12.499999999999972, 'suffix=lorg': 0.1, 'suffix=e.ai': 0.5, 'suffix=l.co': 0.7, 'suffix=scom': 0.7999999999999999, 'suffix=..au': 1.9000000000000006, 'suffix=s.io': 2.400000000000001, 'suffix=dio': 0.2, 'suffix=rdio': 0.2, 'suffix=sedu': 0.7999999999999999, 'suffix=t.me': 0.30000000000000004, 'suffix=e.co': 0.6, 'suffix=a.ai': 0.6, 'suffix=s.me': 2.600000000000001, 'suffix=ecom': 0.2, 'suffix=rme': 0.5, 'suffix=erme': 0.4, 'suffix=nedu': 0.7, 'suffix=n.io': 1.8000000000000005, 'suffix=o.io': 0.30000000000000004, 'suffix=nio': 0.2, 'suffix=onio': 0.2, 'suffix=l.ai': 0.7, 'suffix=yedu': 0.1, 'suffix=a.me': 1.0999999999999999, 'suffix=zedu': 0.30000000000000004, 'suffix=r.ai': 0.9999999999999999, 'suffix=sorg': 0.5, 'suffix=h.co': 0.5, 'suffix=t.ai': 0.2, 'suffix=zco': 0.2, 'suffix=ezco': 0.2, 'suffix=..co': 1.4000000000000001, 'suffix=y.me': 0.30000000000000004, 'suffix=rio': 0.7, 'suffix=erio': 0.30000000000000004, 'suffix=l.io': 0.8999999999999999, 'suffix=g.me': 0.2, 'suffix=t.co': 0.2, 'suffix=p.co': 0.2, 'suffix=g.io': 0.1, 'suffix=nai': 0.4, 'suffix=anai': 0.2, 'suffix=sio': 0.30000000000000004, 'suffix=isio': 0.2, 'suffix=snet': 0.4, 'suffix=orio': 0.2, 'suffix=m.io': 0.4, 'suffix=z.ai': 0.8999999999999999, 'suffix=sme': 0.7, 'suffix=isme': 0.2, 'suffix=lco': 0.30000000000000004, 'suffix=llco': 0.30000000000000004, 'suffix=mai': 0.1, 'suffix=amai': 0.1, 'suffix=b.co': 0.1, 'suffix=pio': 0.1, 'suffix=rpio': 0.1, 'suffix=nme': 0.4, 'suffix=inme': 0.1, 'suffix=e.io': 1.3, 'suffix=ycom': 0.30000000000000004, 'suffix=s.co': 1.6000000000000003, 'suffix=ocom': 0.2, 'suffix=yai': 0.1, 'suffix=hyai': 0.1, 'suffix=eedu': 0.5, 'suffix=kedu': 0.6, 'suffix=aco': 0.1, 'suffix=laco': 0.1, 'suffix=onai': 0.2, 'suffix=r.me': 0.8999999999999999, 'suffix=tco': 0.2, 'suffix=ltco': 0.2, 'suffix=k.me': 0.1, 'suffix=g.co': 0.30000000000000004, 'suffix=a.co': 0.1, 'suffix=oedu': 0.30000000000000004, 'suffix=eco': 0.30000000000000004, 'suffix=eeco': 0.2, 'suffix=l.me': 0.2, 'suffix=msme': 0.30000000000000004, 'suffix=torg': 0.1, 'suffix=k.co': 0.30000000000000004, 'suffix=orme': 0.1, 'suffix=hai': 0.30000000000000004, 'suffix=thai': 0.30000000000000004, 'suffix=lsme': 0.2, 'suffix=rorg': 0.2, 'suffix=nco': 0.7999999999999999, 'suffix=onco': 0.4, 'suffix=gedu': 0.1, 'suffix=teco': 0.1, 'suffix=rco': 0.4, 'suffix=erco': 0.4, 'suffix=o.co': 0.2, 'suffix=k.io': 0.4, 'suffix=rsai': 0.2, 'suffix=rrio': 0.2, 'suffix=hnet': 0.4, 'suffix=dai': 0.30000000000000004, 'suffix=ndai': 0.30000000000000004, 'suffix=tai': 0.1, 'suffix=ntai': 0.1, 'suffix=rai': 0.2, 'suffix=erai': 0.2, 'suffix=zcom': 0.1, 'suffix=d.me': 0.6, 'suffix=wai': 0.1, 'suffix=awai': 0.1, 'suffix=zai': 0.1, 'suffix=ezai': 0.1, 'suffix=y.co': 0.7999999999999999, 'suffix=ome': 0.30000000000000004, 'suffix=some': 0.1, 'suffix=gio': 0.1, 'suffix=ngio': 0.1, 'suffix=hio': 0.1, 'suffix=chio': 0.1, 'suffix=h.ai': 0.6, 'suffix=yorg': 0.1, 'suffix=rome': 0.1, 'suffix=onme': 0.30000000000000004, 'suffix=t.io': 0.2, 'suffix=h.me': 0.1, 'suffix=wnco': 0.30000000000000004, 'suffix=msai': 0.2, 'suffix=aai': 0.2, 'suffix=laai': 0.2, 'suffix=zio': 0.2, 'suffix=ezio': 0.2, 'suffix=enet': 0.2, 'suffix=esai': 0.30000000000000004, 'suffix=ksai': 0.1, 'suffix=h.io': 0.30000000000000004, 'suffix=znet': 0.1, 'suffix=g.ai': 0.4, 'suffix=reai': 0.1, 'suffix=ncom': 0.2, 'suffix=o.me': 0.30000000000000004, 'suffix=kcom': 0.2, 'suffix=sco': 0.6, 'suffix=esco': 0.1, 'suffix=nsai': 0.1, 'suffix=isco': 0.1, 'suffix=osco': 0.1, 'suffix=eio': 0.1, 'suffix=veio': 0.1, 'suffix=lsco': 0.1, 'suffix=k.ai': 0.1, 'suffix=redu': 0.1, 'suffix=thme': 0.1, 'suffix=nsco': 0.1, 'suffix=rnet': 0.1, 'suffix=wsco': 0.1, 'suffix=yio': 0.2, 'suffix=eyio': 0.2, 'suffix=zorg': 0.1, 'suffix=osai': 0.1, 'suffix=dorg': 0.1, 'suffix=eme': 0.1, 'suffix=oeme': 0.1, 'suffix=x.io': 0.1, 'suffix=alio': 0.1, 'suffix=w.co': 0.1, 'suffix=dnet': 0.1, 'suffix=lcom': 0.1, 'suffix=iai': 0.1, 'suffix=liai': 0.1, 'suffix=asio': 0.1, 'suffix=nome': 0.1, 'suffix=rcom': 0.1, 'suffix=inco': 0.1}
print(f"Final results:: w = {WEIGHTS}")

# Use the trained model to predict the labels of the test emails

def test_email(email, weights):
    # A. Convert string to vector
    x = get_email_feature_templates(email)

    # B. Calculate Score (Dot Product)
    z = sparse_dot_product(weights, x)

    # C. Decision Rule
    probability = sigmoid(z)

    return probability, z


print(f"{'Email':<20} | {'Score':<6} | {'Probability'}")
print("-" * 500)

for e in test_emails:
    prob, score = test_email(e, WEIGHTS)

    # Format as percentage
    prob_pct = f"{prob * 100:.1f}%"

    print(f"{e} | {score} | {prob_pct}")

