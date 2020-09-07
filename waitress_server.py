from waitress import serve
import agent_api
#choose the port as you like
serve(agent_api.app, host='0.0.0.0', port=5000)