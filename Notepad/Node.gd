extends Node

# Called when the node enters the scene tree for the first time.
func _ready():
	var save_game = File.new()
	if not save_game.file_exists('user://textdata'):
		return

	save_game.open('user://textdata', File.READ)

	$Notepad.text = save_game.get_line()

	save_game.close()

func _on_Notepad_text_changed():
	var save_game = File.new()
	save_game.open('user://textdata', File.WRITE)

	save_game.store_line($Notepad.text)

	save_game.close()
