import tensorflow as tf
import numpy as np
import einops

class SeqEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, max_length, depth):
    super().__init__()
    self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)

    self.token_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=depth,
        mask_zero=True)

    self.add = tf.keras.layers.Add()

  def call(self, seq):
    seq = self.token_embedding(seq) # (batch, seq, depth)

    x = tf.range(tf.shape(seq)[1])  # (seq)
    x = x[tf.newaxis, :]  # (1, seq)
    x = self.pos_embedding(x)  # (1, seq, depth)

    return self.add([seq,x])

class CausalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    # Use Add instead of + so the keras mask propagates through.
    self.add = tf.keras.layers.Add()
    self.layernorm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    attn = self.mha(query=x, value=x,
                    use_causal_mask=True)
    x = self.add([x, attn])
    return self.layernorm(x)

class CrossAttention(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.add = tf.keras.layers.Add()
    self.layernorm = tf.keras.layers.LayerNormalization()

  # x from self-attention, y from encoder
  def call(self, x, y, **kwargs):
    attn, attention_scores = self.mha(
             query=x, value=y,
             return_attention_scores=True)

    self.last_attention_scores = attention_scores

    x = self.add([x, attn])
    return self.layernorm(x)

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, units, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(units=2*units, activation='relu'),
        tf.keras.layers.Dense(units=units),
        tf.keras.layers.Dropout(rate=dropout_rate),
    ])

    self.layernorm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = x + self.seq(x)
    return self.layernorm(x)

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, units, num_heads=1, dropout_rate=0.1):
    super().__init__()

    self.self_attention = CausalSelfAttention(num_heads=num_heads,
                                              key_dim=units,
                                              dropout=dropout_rate)
    self.cross_attention = CrossAttention(num_heads=num_heads,
                                          key_dim=units,
                                          dropout=dropout_rate)
    self.ff = FeedForward(units=units, dropout_rate=dropout_rate)


  def call(self, inputs, training=False):
    # in_seq : Encoded image, out_seq : Embedded text
    in_seq, out_seq = inputs

    # Text input
    out_seq = self.self_attention(out_seq)

    out_seq = self.cross_attention(out_seq, in_seq)

    self.last_attention_scores = self.cross_attention.last_attention_scores

    out_seq = self.ff(out_seq)

    return out_seq

#@title
class TokenOutput(tf.keras.layers.Layer):
  def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
    super().__init__()

    self.dense = tf.keras.layers.Dense(
        units=5000, **kwargs)
    self.tokenizer = tokenizer
    self.banned_tokens = banned_tokens

    self.bias = None
    with open("output_layer_bias.npy", "rb") as f:
      self.bias = np.load(f)

  def call(self, x):
    x = self.dense(x)
    return x + self.bias

output_layer = TokenOutput(None, banned_tokens=('', '[UNK]', '[START]'))

class Captioner(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,
               units=256, max_length=50, num_heads=1, dropout_rate=0.1):
    super().__init__()
    self.feature_extractor = feature_extractor
    self.tokenizer = tokenizer

    self.seq_embedding = SeqEmbedding(
        vocab_size=5000,
        depth=units,
        max_length=max_length)

    self.decoder_layers = [
        DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
        for n in range(num_layers)]

    self.output_layer = output_layer

  def get_prunable_weights(self):
    return [self.feature_extractor.weights]

@Captioner.add_method
def call(self, inputs):
  image, txt = inputs

  # Flatten the feature map
  image = einops.rearrange(image, 'b h w c -> b (h w) c')

  txt = self.seq_embedding(txt)

  # Look at the image
  for dec_layer in self.decoder_layers:
      txt = dec_layer(inputs=(image, txt))

  txt = self.output_layer(txt)

  return txt

def load_and_get_model():
  model = Captioner(None, feature_extractor=None, output_layer=output_layer, units=256, dropout_rate=0.5, num_layers=2, num_heads=2)
  model.load_weights("decoder_weights")
  return model

def inference(model, data, vocabs):
  data = tf.convert_to_tensor(data, dtype=tf.float32)
  tokens = tf.convert_to_tensor([[3]], dtype=tf.int64) # [START]
  for n in range(50):
    preds = model((data, tokens)).numpy()
    preds = preds[:,-1,:]
    nxt = tf.argmax(preds, axis=-1)[:, tf.newaxis]
    tokens = tf.concat([tokens, nxt], axis=1)
    if int(nxt.numpy().tolist()[0][0]) == 4:
      break # [END]
  result = tokens.numpy().tolist()[0][1:-1]
  return " ".join([vocabs[tok] for tok in result])
