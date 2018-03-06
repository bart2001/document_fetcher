# document_fetcher

public static Map<String, Boolean> makeMap(Map<String, Boolean> linkedMap) {
		Map<String, Boolean> returnMap = new LinkedHashMap<String, Boolean>();
		for (String key : linkedMap.keySet()) {
			boolean value = linkedMap.get(key);
			if (value) {
				returnMap.put(key, value);	
			} else {
				try {
					List<String> childList = childMap.get(key);
					returnMap.put(key, true);
					for (String child : childList) {
						returnMap.put(child, false);
					}
				} catch (Exception e) {
					returnMap.put(key, true);
				}
			}
		}
		return returnMap;
	}
