# This function contains function definitions for OpenAI function calling

function_list = [
    {
        "name": "track_marketing_symbol",
        "description": "This function is used to track the news related to marketing symbol. Only call this function when user exactly asked to track specific marketing symbol's activity to system.",
        "parameters": {  # parameters
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Marketing symbol to track. e.g microsoft. While extract the symbol, please correct spell mistakes. e.g micrsoft -> microsoft",
                    },
                },
            "required": ["symbol"],
        },
    },
    {
        "name": "stop_tracking",
        "description": "This function is used to stop tracking a marketing symbol. Only call this function when user exactly asked to stop track specific marketing symbol's activity to system.",
        "parameters": {  # parameters
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Marketing symbol to stop track. e.g microsoft. While extract the symbol, please correct spell mistakes. e.g micrsoft -> microsoft",
                    },
                },
            "required": ["symbol"],
        },
    }
]