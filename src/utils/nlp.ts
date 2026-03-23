/**
 * Simple NLP utility for preprocessing text.
 * In a production environment, this would be handled by a more robust library
 * or as part of the model pipeline.
 */

export const tokenize = (text: string): string[] => {
  return text.toLowerCase().match(/\b(\w+)\b/g) || [];
};

export const removeStopwords = (tokens: string[]): string[] => {
  const stopwords = new Set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
  ]);
  return tokens.filter(token => !stopwords.has(token));
};

export const lemmatize = (tokens: string[]): string[] => {
  // Basic suffix stripping for demonstration
  return tokens.map(token => {
    if (token.endsWith('ing')) return token.slice(0, -3);
    if (token.endsWith('ed')) return token.slice(0, -2);
    if (token.endsWith('s') && !token.endsWith('ss')) return token.slice(0, -1);
    return token;
  });
};

export const preprocess = (text: string) => {
  const tokens = tokenize(text);
  const filtered = removeStopwords(tokens);
  const lemmatized = lemmatize(filtered);
  return {
    tokens,
    filtered,
    lemmatized
  };
};
