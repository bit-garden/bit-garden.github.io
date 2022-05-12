extends Node

var current_file := ''

const menu_items = ['New', 'Open']

# Called when the node enters the scene tree for the first time.
func _ready():
	for m in menu_items:
		$MenuButton.get_popup().add_item(m)

	$MenuButton.get_popup().connect("id_pressed", self, "_on_item_pressed")

func _on_item_pressed(id):
	var item = menu_items[id]

	if item == 'New':
		$SaveDialog.current_dir = 'user://'
		$SaveDialog.popup()
	elif item == 'Open':
		$OpenDialog.current_dir = 'user://'
		$OpenDialog.popup()

func _on_Notepad_text_changed():
	if current_file == '':
		return

	if $Notepad.text.strip_edges() == '':
		var dir = Directory.new()
		dir.remove(current_file)
		return

	var save_game := File.new()
	save_game.open(current_file.strip_edges(), File.WRITE)

	save_game.store_string($Notepad.text)

	save_game.close()

func _on_FileDialog_file_selected(path):
	current_file = path

	var save_game = File.new()
	if not save_game.file_exists(path):
		return

	save_game.open(path, File.READ)

	$Notepad.text = ''

	while not save_game.eof_reached():
		$Notepad.text += save_game.get_line() + '\n'

	save_game.close()


func _on_SaveDialog_file_selected(path):
	current_file = path
	$Notepad.text = ''
